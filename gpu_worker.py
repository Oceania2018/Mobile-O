"""GPU Worker: Redis queue → HTTP API bridge for Mobile-O image understanding.

Receives tasks from a Redis List (BRPOP), forwards them to the running
api_image_understanding.py FastAPI server via HTTP, and publishes results
back via Redis Pub/Sub – matching the protocol defined in
docs/gpu-rpc-integration.md.

The model is loaded only once (inside api_image_understanding.py), so both
processes can run simultaneously without doubling GPU memory usage.

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
    # Terminal 1 – model server
    python api_image_understanding.py --model_path checkpoints/final_merged_model_23620

    # Terminal 2 – Redis worker (no GPU required here)
    python gpu_worker.py

    # Azure Redis (SSL):
    REDIS_ACCESS_KEY=<key> python gpu_worker.py \\
        --redis_host rpc.centralus.redis.azure.net --redis_port 10000
"""

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import redis
from redis.cluster import RedisCluster

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--api_url", type=str, default=os.environ.get("API_URL", "http://localhost:8010"),
                    help="Base URL of the running api_image_understanding.py server")
parser.add_argument("--api_timeout", type=float, default=60.0,
                    help="HTTP timeout in seconds when calling the API server")
parser.add_argument("--redis_host", type=str, default=os.environ.get("REDIS_HOST", "rpc.centralus.redis.azure.net"))
parser.add_argument("--redis_port", type=int, default=int(os.environ.get("REDIS_PORT", "10000")))
parser.add_argument("--redis_db",   type=int, default=0)
parser.add_argument("--queue_name", type=str, default="ai_tasks:image_understand",
                    help="Redis List key to BRPOP from (must match C# TaskQueueName)")
parser.add_argument("--result_channel_prefix", type=str, default="result",
                    help="Pub/Sub channel prefix; result published to <prefix>:<taskId>")
# Auto-start API server options
parser.add_argument("--auto_start_api", action="store_true",
                    help="Automatically launch api_image_understanding.py as a subprocess")
parser.add_argument("--model_path", type=str, default="checkpoints/final_merged_model_23620",
                    help="Model path forwarded to api_image_understanding.py (only used with --auto_start_api)")
parser.add_argument("--api_ready_timeout", type=float, default=300.0,
                    help="Seconds to wait for the API server to become healthy (only used with --auto_start_api)")
parser.add_argument("--redis_cluster", action="store_true",
                    help="Connect using RedisCluster client (required when Redis runs in Cluster mode, e.g. Azure Redis Cluster)")
args = parser.parse_args()

REDIS_ACCESS_KEY = os.environ.get("REDIS_ACCESS_KEY", "")
REDIS_SSL = bool(REDIS_ACCESS_KEY)

# ---------------------------------------------------------------------------
# Task handler – delegates to api_image_understanding.py via HTTP
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def _decode_b64_entry(entry: str) -> bytes:
    """Decode a raw base64 string or data URI to bytes."""
    if entry.startswith("data:"):
        _, b64data = entry.split(",", 1)
    else:
        b64data = entry
    return base64.b64decode(b64data)


def _fetch_url_bytes(url: str) -> bytes:
    """Fetch image bytes from an HTTP/HTTPS URL synchronously."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
    response.raise_for_status()
    return response.content


def _resolve_image_entry(args: tuple) -> tuple:
    """Fetch or decode a single image entry; returns (index, bytes)."""
    i, entry = args
    entry = str(entry).strip()
    if _is_url(entry):
        return i, _fetch_url_bytes(entry)
    return i, _decode_b64_entry(entry)


def handle_image_understand(task_args: dict) -> dict:
    """Handler for task "image-understand".

    Forwards the request to the running FastAPI server at args.api_url.
    Accepted keys in task_args:
        images         – list of image sources: base64, data URI, or HTTP(S) URL (required)
        mode           – "caption" | "description" | "prompt" (default: "caption")
        text           – custom prompt text, required when mode="prompt"
        temperature    – float (optional)
        max_new_tokens – int   (optional)
    """
    images_list = task_args.get("images")
    if not images_list:
        return {"status": "error", "error": "Missing 'images' in args. Provide a list of base64, data URI, or URL strings."}
    if not isinstance(images_list, list):
        images_list = [images_list]  # accept a bare string for convenience

    # Resolve each entry to bytes in parallel (URL downloads + base64 decodes)
    n = len(images_list)
    resolved: dict[int, bytes] = {}
    errors: dict[int, dict] = {}
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
        return errors[min(errors)]  # return the first error (lowest index)

    files = [("images", (f"image_{i}.jpg", io.BytesIO(resolved[i]), "image/jpeg")) for i in range(n)]

    # Optional scalar fields
    form_data: dict = {}
    for key in ("mode", "text", "temperature", "max_new_tokens"):
        val = task_args.get(key)
        if val is not None:
            form_data[key] = str(val)

    try:
        with httpx.Client(timeout=args.api_timeout) as client:
            response = client.post(
                f"{args.api_url}/understand",
                files=files,
                data=form_data,
            )
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except httpx.ConnectError:
        msg = (f"Cannot connect to API server at {args.api_url}. "
               "Is api_image_understanding.py running?")
        print(f"  [ConnectError] {msg}")
        return {"status": "error", "error": msg}
    except httpx.HTTPStatusError as exc:
        msg = f"API error {exc.response.status_code}: {exc.response.text}"
        print(f"  [HTTP error] {msg}")
        return {"status": "error", "error": msg}
    except httpx.TimeoutException:
        msg = f"API timeout after {args.api_timeout}s  (url={args.api_url}/understand)"
        print(f"  [Timeout] {msg}")
        return {"status": "error", "error": msg}
    except Exception as exc:
        import traceback
        print(f"  [Exception in handle_image_understand] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(exc)}


# Registry – add new handlers here as the system grows
TASK_HANDLERS: dict = {
    "image-understand": handle_image_understand,
}


# ---------------------------------------------------------------------------
# Optional: auto-start api_image_understanding.py
# ---------------------------------------------------------------------------
_api_proc: "subprocess.Popen | None" = None

def _start_api_server() -> None:
    """Launch api_image_understanding.py as a child process and wait until healthy."""
    global _api_proc
    from urllib.parse import urlparse

    parsed = urlparse(args.api_url)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8010

    cmd = [
        sys.executable, "api_image_understanding.py",
        "--model_path", args.model_path,
        "--host", host,
        "--port", str(port),
    ]
    print(f"[auto-start] Launching: {' '.join(cmd)}")
    _api_proc = subprocess.Popen(cmd)

    health_url = f"{args.api_url}/health"
    deadline = time.monotonic() + args.api_ready_timeout
    poll_interval = 3.0
    print(f"[auto-start] Waiting for API server to become healthy at {health_url} "
          f"(timeout={args.api_ready_timeout}s) ...")
    while time.monotonic() < deadline:
        # Check if the subprocess crashed early
        if _api_proc.poll() is not None:
            print(f"[auto-start] API process exited unexpectedly (returncode={_api_proc.returncode}). Aborting.")
            sys.exit(1)
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(health_url)
            if resp.status_code == 200:
                print(f"[auto-start] API server is ready. ({resp.json()})")
                return
        except httpx.TransportError:
            pass  # not up yet
        time.sleep(poll_interval)

    print(f"[auto-start] API server did not become healthy within {args.api_ready_timeout}s. Aborting.")
    _api_proc.terminate()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Redis worker loop
# ---------------------------------------------------------------------------
def _connect_redis():
    if args.redis_cluster:
        # Azure Redis Cluster gossips internal IPs (e.g. 20.15.156.76:8501) to
        # clients.  Those IPs are not externally reachable and the TLS cert is
        # only valid for the public hostname.  address_remap redirects every
        # node address back to the public endpoint so all traffic goes through
        # the one host whose cert IS valid.
        import ssl
        public_addr = (args.redis_host, args.redis_port)
        
        # Create SSL context that skips hostname verification for internal IPs
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
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


_running = True

def _on_signal(signum, _frame):
    global _running
    print(f"\nReceived signal {signum}, shutting down ...")
    _running = False
    if _api_proc is not None and _api_proc.poll() is None:
        print("[auto-start] Terminating API subprocess ...")
        _api_proc.terminate()

signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def run_worker():
    if args.auto_start_api:
        _start_api_server()

    r = _connect_redis()
    print(f"Connected to Redis {args.redis_host}:{args.redis_port} (ssl={REDIS_SSL})")
    print(f"API server  : {args.api_url}  (timeout={args.api_timeout}s)")
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
            status = result.get('status')
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


if __name__ == "__main__":
    run_worker()
