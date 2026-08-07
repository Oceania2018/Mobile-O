"""GPU Worker: serves the Mobile-O image-understanding model over an
OpenAI-compatible HTTP API for the llm-inference-daemon.

Loads the model once and runs the OpenAI-compatible server (see openai_server.py)
in the foreground. Tasks arrive as OpenAI chat requests on --http_port
(default 8400); the llm-inference-daemon forwards them and routes by model id.

Usage:
    python gpu_worker.py --model_path checkpoints/final_merged_model_23620 \\
        --http_port 8400 --model_id mobile-o-0.5b
"""

from argparse import ArgumentParser

from mobileo_engine import MobileOEngine
from openai_server import serve

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = ArgumentParser()
# Model
parser.add_argument("--model_path", type=str, default="checkpoints/final_merged_model_23620")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--mode", type=str, choices=["caption", "description", "prompt"], default="caption",
                    help="Default inference mode when a request does not specify one")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--max_batch_images", type=int, default=8,
                    help="Max images processed per GPU batch. Larger requests are split "
                         "into chunks of this size to bound peak VRAM; the CUDA cache is "
                         "released after each task to keep the high-water mark from creeping up.")
parser.add_argument("--enable_image_generation", action="store_true",
                    help="Load the Sana DiT + VAE image-generation stack (~1.7 GiB VRAM). "
                         "Off by default: this server only exposes image understanding, so "
                         "the generation modules would sit idle in VRAM.")
# OpenAI-compatible HTTP endpoint
parser.add_argument("--http_port", type=int, default=8400,
                    help="Port to serve the OpenAI-compatible HTTP API on (for the "
                         "llm-inference-daemon).")
parser.add_argument("--http_host", type=str, default="0.0.0.0",
                    help="Bind host for the OpenAI HTTP endpoint.")
parser.add_argument("--model_id", type=str, default="mobile-o-0.5b",
                    help="Model id advertised at /v1/models; must match the daemon's "
                         "model.servers.json 'Models' entry.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
engine = MobileOEngine(
    args.model_path,
    device=args.device,
    default_mode=args.mode,
    default_temperature=args.temperature,
    default_max_new_tokens=args.max_new_tokens,
    max_batch_images=args.max_batch_images,
    understand_only=not args.enable_image_generation,
)


if __name__ == "__main__":
    # serve() blocks until the process receives SIGINT/SIGTERM (uvicorn installs
    # its own handlers and shuts down gracefully); release the model afterwards.
    try:
        serve(engine, model_id=args.model_id, host=args.http_host, port=args.http_port)
    finally:
        engine.shutdown()
