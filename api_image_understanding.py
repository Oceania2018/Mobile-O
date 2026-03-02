"""
REST API for Mobile-O image understanding inference.

Usage:
    python api_image_understanding.py --model_path checkpoints/final_merged_model_23620

    # Then send requests:
    # Single image, caption mode (default)
    curl -X POST http://localhost:8010/understand -F "images=@assets/cute_cat.png"

    # Multiple images, description mode
    curl -X POST http://localhost:8010/understand \
        -F "images=@assets/img1.png" -F "images=@assets/img2.png" \
        -F "mode=description"

    # Multiple images, custom prompt
    curl -X POST http://localhost:8010/understand \
        -F "images=@assets/img1.png" -F "images=@assets/img2.png" \
        -F "mode=prompt" -F "text=What objects are in these images?"
"""

import io
import asyncio
import time
from typing import List
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from argparse import ArgumentParser

import warnings
from transformers import AutoTokenizer
from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mobileo.model import mobileoForInferenceLM
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/final_merged_model_23620")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8010)
parser.add_argument(
    "--mode",
    type=str,
    choices=["caption", "description", "prompt"],
    default="caption",
    help="caption: Caption the image | description: Describe the image | prompt: provide custom text via --text",
)

parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                    help="Device to run inference on: cuda or cpu")
args = parser.parse_args()

MODE_PROMPTS = {
    "caption": "Caption this image in under 16 words.",
    "description": "Describe the image in under 32 words.",
}

# ---------------------------------------------------------------------------
# Global state
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

image_processor = model.get_vision_tower().image_processor

# ---------------------------------------------------------------------------
# Pre-compile the fixed prompt (conversation template + tokenization)
# ---------------------------------------------------------------------------
model.generation_config.pad_token_id = tokenizer.pad_token_id

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

_executor = ThreadPoolExecutor(max_workers=1)
_gpu_lock = asyncio.Lock()

# Separate thread pool for CPU-bound image decoding (PIL), sized to available CPUs
import os as _os
_decode_executor = ThreadPoolExecutor(max_workers=min((_os.cpu_count() or 4), 8))


def _decode_pil(image_bytes: bytes) -> Image.Image:
    """Open and convert image bytes to an RGB PIL Image (runs in thread pool)."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
print("Warmup complete. Server ready.")


# ---------------------------------------------------------------------------
# Core inference (runs inside the thread-pool, not the event loop)
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
        batch_size = len(images)
        batched_input_ids = input_ids.repeat(batch_size, 1)
        batched_images = image_tensors.to(DTYPE)
        with torch.inference_mode():
            output_ids = model.generate(batched_input_ids, images=batched_images, **gen_kwargs)
        return [s.strip() for s in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    # Fallback: anyres returned tensors with different shapes (different patch counts).
    # Resize all images to the processor's fixed crop size so shapes unify, then retry batch.
    target_size = image_processor.crop_size["height"]
    resized_images = [img.resize((target_size, target_size), Image.BICUBIC) for img in images]
    image_tensors = process_images(resized_images, image_processor, model.config)

    if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
        batch_size = len(images)
        batched_input_ids = input_ids.repeat(batch_size, 1)
        batched_images = image_tensors.to(DTYPE)
        with torch.inference_mode():
            output_ids = model.generate(batched_input_ids, images=batched_images, **gen_kwargs)
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
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False)
    _decode_executor.shutdown(wait=False)

app = FastAPI(title="Mobile-O Image Understanding API", lifespan=lifespan)


VALID_MODES = {"caption", "description", "prompt"}

@app.post("/understand")
async def understand_image(
    images: List[UploadFile] = File(..., description="One or more image files"),
    temperature: float = Form(default=None, description="Sampling temperature"),
    max_new_tokens: int = Form(default=None, description="Max new tokens"),
    mode: str = Form(default=None, description="caption | description | prompt (default: server --mode)"),
    text: str = Form(default=None, description="Custom prompt text, required when mode=prompt"),
):
    temp = temperature if temperature is not None else args.temperature
    tokens = max_new_tokens if max_new_tokens is not None else args.max_new_tokens

    effective_mode = mode if mode is not None else args.mode
    if effective_mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{effective_mode}'. Must be one of: {sorted(VALID_MODES)}.")

    # Resolve input_ids
    if effective_mode == "prompt":
        if not text:
            raise HTTPException(status_code=400, detail="'text' form field is required when mode=prompt.")
        input_ids = _build_input_ids(text)
    else:
        input_ids = _cached_input_ids_map[effective_mode]

    # Step 1: read all uploaded files concurrently (async I/O)
    t_img_start = time.perf_counter()
    all_bytes: List[bytes] = await asyncio.gather(*[u.read() for u in images])

    # Step 2: decode PIL images in parallel via the decode thread pool
    loop = asyncio.get_event_loop()
    decode_results = await asyncio.gather(
        *[loop.run_in_executor(_decode_executor, _decode_pil, b) for b in all_bytes],
        return_exceptions=True,
    )
    pil_images: List[Image.Image] = []
    for img_or_exc, upload in zip(decode_results, images):
        if isinstance(img_or_exc, Exception):
            raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}")
        pil_images.append(img_or_exc)
    t_img_elapsed = round(time.perf_counter() - t_img_start, 3)
    print(f"image read+decode: {t_img_elapsed}s  (n={len(pil_images)})")

    t_infer_start = time.perf_counter()
    async with _gpu_lock:
        results = await loop.run_in_executor(
            _executor, _run_inference, pil_images, temp, tokens, input_ids
        )
    t_infer_elapsed = round(time.perf_counter() - t_infer_start, 3)
    print(f"inference: {t_infer_elapsed}s  results={results}")

    return JSONResponse(content={
        "responses": results,
        "elapsed_seconds": round(t_img_elapsed + t_infer_elapsed, 3),
        "image_decode_seconds": t_img_elapsed,
        "inference_seconds": t_infer_elapsed,
    })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_path": args.model_path,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

