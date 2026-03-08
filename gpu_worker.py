"""GPU Worker: Redis queue → model inference for Mobile-O image understanding.

Receives tasks from a Redis List (BRPOP), runs model inference directly,
and publishes results back via Redis Pub/Sub.

Incoming task message (JSON pushed by C# RedisRpcService):
    {
        "taskId": "<uuid>",
        "task":   "image-understand",
        "args":   {
            "images":         ["<base64>", "data:image/png;base64,...", "https://..."],
            "mode":           "caption" | "description" | "prompt",
            "text":           "<custom prompt, required when mode=prompt>",
            "temperature":    0.0,
            "max_new_tokens": 64
        }
    }

Result message published to  result:<taskId>:
    { "status": "success", "data": { "responses": ["..."], "elapsed_seconds": ... } }
    { "status": "error",   "error": "..." }

Usage:
    python gpu_worker.py --model_path checkpoints/final_merged_model_23620

    # Azure Redis (SSL):
    REDIS_ACCESS_KEY=<key> python gpu_worker.py \\
        --model_path checkpoints/final_merged_model_23620 \\
        --redis_host rpc.centralus.redis.azure.net --redis_port 10000
"""

import base64
import io
import json
import os
import signal
import time
import warnings
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import httpx
import redis
import torch
from PIL import Image
from redis.cluster import RedisCluster
from transformers import AutoTokenizer

from mobileo.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                                DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                                IMAGE_TOKEN_INDEX)
from mobileo.conversation import conv_templates
from mobileo.mm_utils import process_images, tokenizer_image_token
from mobileo.model import mobileoForInferenceLM
from mobileo.utils import disable_torch_init

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = ArgumentParser()
# Model
parser.add_argument("--model_path", type=str, default="checkpoints/final_merged_model_23620")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--mode", type=str, choices=["caption", "description", "prompt"], default="caption",
                    help="Default inference mode when task does not specify one")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_new_tokens", type=int, default=64)
# Redis
parser.add_argument("--redis_host", type=str, default=os.environ.get("REDIS_HOST", "rpc.centralus.redis.azure.net"))
parser.add_argument("--redis_port", type=int, default=int(os.environ.get("REDIS_PORT", "10000")))
parser.add_argument("--redis_db",   type=int, default=0)
parser.add_argument("--queue_name", type=str, default="ai_tasks:image_understand",
                    help="Redis List key to BRPOP from (must match C# TaskQueueName)")
parser.add_argument("--result_channel_prefix", type=str, default="result",
                    help="Pub/Sub channel prefix; result published to <prefix>:<taskId>")
parser.add_argument("--redis_cluster", action="store_true", default=True,
                    help="Connect using RedisCluster client (required for Azure Redis Cluster)")
args = parser.parse_args()

REDIS_ACCESS_KEY = os.environ.get("REDIS_ACCESS_KEY", "")
REDIS_SSL = bool(REDIS_ACCESS_KEY)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
DEVICE = args.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    print("WARNING: CUDA not available, falling back to CPU.")
    DEVICE = "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Loading model from {args.model_path} (device={DEVICE}, dtype={DTYPE}) ...")
disable_torch_init()
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
model = mobileoForInferenceLM.from_pretrained(
    args.model_path,
    low_cpu_mem_usage=True,
    torch_dtype=DTYPE,
    device_map=DEVICE if DEVICE == "cpu" else "auto",
)
mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
if mm_use_im_patch_token:
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
model.eval()
model.generation_config.pad_token_id = tokenizer.pad_token_id

image_processor = model.get_vision_tower().image_processor

MODE_PROMPTS = {
    "caption":     "Caption this image in under 16 words.",
    "description": "Describe the image in under 32 words.",
}


def _build_input_ids(text: str):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    return (
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(DEVICE)
    )


print("Pre-compiling fixed prompts ...")
_cached_input_ids_map = {}
for _m, _p in MODE_PROMPTS.items():
    _cached_input_ids_map[_m] = _build_input_ids(_p)
    print(f"  [{_m}] pre-compiled: {_cached_input_ids_map[_m].shape[1]} tokens")

_decode_executor = ThreadPoolExecutor(max_workers=min((os.cpu_count() or 4), 8))

# Warmup pass
print("Running warmup pass ...")
_warmup_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
_warmup_img = process_images([_warmup_image], image_processor, model.config)[0]
with torch.inference_mode():
    model.generate(
        _cached_input_ids_map["caption"],
        images=_warmup_img.unsqueeze(0).to(DTYPE),
        do_sample=False,
        num_beams=1,
        max_new_tokens=8,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
del _warmup_image, _warmup_img
print("Model ready.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _run_inference(images: List[Image.Image], temperature: float, max_new_tokens: int, input_ids) -> List[str]:
    gen_kwargs = dict(
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    image_tensors = process_images(images, image_processor, model.config)

    # Batch path: process_images returned a single [N, C, H, W] tensor
    if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
        batched_input_ids = input_ids.repeat(len(images), 1)
        with torch.inference_mode():
            output_ids = model.generate(batched_input_ids, images=image_tensors.to(DTYPE), **gen_kwargs)
        return [s.strip() for s in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    # Fallback: anyres returned tensors with different shapes (different patch counts).
    # Resize all images to the processor's fixed crop size so shapes unify, then retry batch.
    target_size = image_processor.crop_size["height"]
    resized_images = [img.resize((target_size, target_size), Image.BICUBIC) for img in images]
    image_tensors = process_images(resized_images, image_processor, model.config)

    if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
        batched_input_ids = input_ids.repeat(len(images), 1)
        with torch.inference_mode():
            output_ids = model.generate(batched_input_ids, images=image_tensors.to(DTYPE), **gen_kwargs)
        return [s.strip() for s in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    # Last resort: loop (should not normally be reached)
    results = []
    tensor_list = image_tensors if isinstance(image_tensors, list) else [image_tensors[i] for i in range(len(images))]
    for img_tensor in tensor_list:
        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=img_tensor.unsqueeze(0).to(DTYPE), **gen_kwargs)
        results.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
    return results

# ---------------------------------------------------------------------------
# Image resolution helpers (base64 / data URI / URL)
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def _decode_b64_entry(entry: str) -> bytes:
    if entry.startswith("data:"):
        _, b64data = entry.split(",", 1)
    else:
        b64data = entry
    return base64.b64decode(b64data)


def _fetch_url_bytes(url: str) -> bytes:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
    response.raise_for_status()
    return response.content


def _resolve_image_entry(entry_tuple: tuple) -> tuple:
    i, entry = entry_tuple
    entry = str(entry).strip()
    if _is_url(entry):
        return i, _fetch_url_bytes(entry)
    return i, _decode_b64_entry(entry)


def _decode_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# ---------------------------------------------------------------------------
# Task handler
# ---------------------------------------------------------------------------
VALID_MODES = {"caption", "description", "prompt"}


def handle_image_understand(task_args: dict) -> dict:
    images_list = task_args.get("images")
    if not images_list:
        return {"status": "error", "error": "Missing 'images' in args. Provide a list of base64, data URI, or URL strings."}
    if not isinstance(images_list, list):
        images_list = [images_list]

    n = len(images_list)
    resolved: dict[int, bytes] = {}
    errors: dict[int, dict] = {}

    t_img_start = time.perf_counter()

    # Resolve entries (URL fetch or base64 decode) in parallel
    with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
        future_to_idx = {
            pool.submit(_resolve_image_entry, (i, entry)): i
            for i, entry in enumerate(images_list)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                idx, data = future.result()
                resolved[idx] = data
            except httpx.HTTPStatusError as exc:
                entry = str(images_list[i]).strip()
                errors[i] = {"status": "error", "error": f"Failed to fetch image URL '{entry}': HTTP {exc.response.status_code}"}
            except Exception as exc:
                errors[i] = {"status": "error", "error": f"Invalid image at images[{i}]: {exc}"}

    if errors:
        return errors[min(errors)]

    # Decode PIL images in parallel
    decode_futures = {_decode_executor.submit(_decode_pil, resolved[i]): i for i in range(n)}
    pil_images: List[Image.Image] = [None] * n
    for future in as_completed(decode_futures):
        i = decode_futures[future]
        try:
            pil_images[i] = future.result()
        except Exception as exc:
            return {"status": "error", "error": f"Invalid image bytes at images[{i}]: {exc}"}

    t_img_elapsed = round(time.perf_counter() - t_img_start, 3)

    # Resolve mode / input_ids
    temperature    = float(task_args.get("temperature", args.temperature))
    max_new_tokens = int(task_args.get("max_new_tokens", args.max_new_tokens))
    mode = task_args.get("mode", args.mode)
    if mode not in VALID_MODES:
        return {"status": "error", "error": f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}."}

    if mode == "prompt":
        text = task_args.get("text", "")
        if not text:
            return {"status": "error", "error": "'text' is required when mode=prompt."}
        input_ids = _build_input_ids(text)
    else:
        input_ids = _cached_input_ids_map[mode]

    t_infer_start = time.perf_counter()
    try:
        responses = _run_inference(pil_images, temperature, max_new_tokens, input_ids)
    except Exception as exc:
        import traceback
        print(f"  [inference error] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(exc)}
    t_infer_elapsed = round(time.perf_counter() - t_infer_start, 3)

    print(f"  image_decode={t_img_elapsed}s  inference={t_infer_elapsed}s  results={responses}")
    return {
        "status": "success",
        "data": {
            "responses": responses,
            "elapsed_seconds": round(t_img_elapsed + t_infer_elapsed, 3),
            "image_decode_seconds": t_img_elapsed,
            "inference_seconds": t_infer_elapsed,
        },
    }


# Registry – add new handlers here as the system grows
TASK_HANDLERS: dict = {
    "image-understand": handle_image_understand,
}

# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------
def _connect_redis():
    if args.redis_cluster:
        # Azure Redis Cluster gossips internal IPs to clients; those IPs are not
        # externally reachable and the TLS cert is only valid for the public hostname.
        # address_remap redirects every node address back to the public endpoint.
        public_addr = (args.redis_host, args.redis_port)
        return RedisCluster(
            host=args.redis_host,
            port=args.redis_port,
            password=REDIS_ACCESS_KEY or None,
            ssl=True,
            ssl_certfile=None,
            ssl_keyfile=None,
            ssl_ca_certs=None,
            ssl_check_hostname=False,
            decode_responses=True,
            skip_full_coverage_check=True,
            address_remap=lambda _addr: public_addr,
        )
    return redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=REDIS_ACCESS_KEY or None,
        ssl=REDIS_SSL,
        ssl_check_hostname=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        ssl_ca_certs=None,
        decode_responses=True,
    )

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_running = True


def _on_signal(signum, _frame):
    global _running
    print(f"\nReceived signal {signum}, shutting down ...")
    _running = False


signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------
def run_worker():
    r = _connect_redis()
    print(f"Connected to Redis {args.redis_host}:{args.redis_port} (ssl={REDIS_SSL})")
    print(f"Listening on: {args.queue_name}")

    while _running:
        try:
            # Block up to 2 s so we can check _running between polls
            item = r.brpop(args.queue_name, timeout=2)
            if item is None:
                continue

            _, raw = item
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] JSON decode failed: {exc} | raw={raw[:200]}")
                continue

            task_id   = payload.get("taskId", "")
            task_name = payload.get("task", "")
            task_args = payload.get("args", {})

            print(f"[{task_id}] task={task_name}  args_keys={list(task_args.keys())}")
            t0 = time.perf_counter()

            handler = TASK_HANDLERS.get(task_name)
            if handler is None:
                result = {"status": "error", "error": f"Unknown task '{task_name}'."}
            else:
                try:
                    result = handler(task_args)
                except Exception as exc:
                    import traceback
                    print(f"[{task_id}] [EXCEPTION] {type(exc).__name__}: {exc}")
                    print(traceback.format_exc())
                    result = {"status": "error", "error": str(exc)}

            elapsed = round(time.perf_counter() - t0, 3)
            status = result.get("status")
            if status == "error":
                print(f"[{task_id}] status={status}  elapsed={elapsed}s  error={result.get('error')}")
            else:
                data = result.get("data", {})
                img_s   = data.get("image_decode_seconds", "n/a")
                infer_s = data.get("inference_seconds", "n/a")
                print(f"[{task_id}] status={status}  elapsed={elapsed}s  "
                      f"(image_decode={img_s}s  inference={infer_s}s)")

            r.publish(f"{args.result_channel_prefix}:{task_id}", json.dumps(result))

        except redis.exceptions.MovedError as exc:
            print(f"[ERROR] MovedError: {exc}. Redis is running in Cluster mode. "
                  "Restart with --redis_cluster to enable the cluster-aware client.")
            time.sleep(3)
            try:
                r = _connect_redis()
            except Exception as re_exc:
                import traceback
                print(f"[ERROR] Reconnect failed ({type(re_exc).__name__}): {re_exc}")
                print(traceback.format_exc())
        except redis.exceptions.ConnectionError as exc:
            print(f"[ERROR] Redis connection lost ({type(exc).__name__}): {exc}. Reconnecting in 3 s ...")
            time.sleep(3)
            try:
                r = _connect_redis()
            except Exception as re_exc:
                import traceback
                print(f"[ERROR] Reconnect failed ({type(re_exc).__name__}): {re_exc}")
                print(traceback.format_exc())
        except Exception as exc:
            import traceback
            print(f"[ERROR] Unexpected error ({type(exc).__name__}): {exc}")
            print(traceback.format_exc())

    print("Worker stopped.")
    _decode_executor.shutdown(wait=False)


if __name__ == "__main__":
    run_worker()
