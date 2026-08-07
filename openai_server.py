"""OpenAI-compatible HTTP front-end for the Mobile-O image-understanding model.

Exposes exactly the two endpoints the llm-inference-daemon expects from any
backend, so Mobile-O can be registered in the daemon's ``model.servers.json``
like any other OpenAI server (no daemon code changes needed):

    GET  /v1/models             -> model listing; also the daemon's health probe
    POST /v1/chat/completions   -> vision chat; image(s) come in as OpenAI
                                   ``image_url`` content parts (base64 data URI,
                                   bare base64, or http(s) URL)

The daemon forwards the request body unchanged and routes by the ``model`` field,
so this server only has to (a) advertise its model id at ``/v1/models`` and
(b) translate an OpenAI chat request into a Mobile-O inference call.

This module does not load a model itself when embedded — it is handed an
already-constructed :class:`mobileo_engine.MobileOEngine`, so ``gpu_worker.py``
keeps a single model copy in the process. It can also run standalone
(``python openai_server.py --model_path ...``) for testing, which loads its own
engine.
"""

import logging
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mobileo_engine import MobileOEngine

# Piggyback on uvicorn's already-configured logger so these lines land in the
# same container log stream as the access log (no extra logging setup needed).
logger = logging.getLogger("uvicorn.error")

# Default model id advertised to the daemon. Must match an entry in the daemon's
# model.servers.json "Models" list (matching is case-insensitive).
DEFAULT_MODEL_ID = "mobile-o-0.5b"


# --- Request shapes (loose — we only read the fields we use) ----------------
class ChatMessage(BaseModel):
    role: str
    # content is either a plain string or a list of OpenAI content parts.
    content: object = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = []
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Non-standard passthrough: lets a caller pin a Mobile-O fixed mode
    # ("caption" | "description"). When absent we derive the mode from the text.
    mode: Optional[str] = None


def _extract_messages(messages: List[ChatMessage]) -> tuple:
    """Pull the system prompt, user text, and image entries out of the messages.

    Honours the standard OpenAI vision shape: a message's content is either a
    plain string or a list of parts, each ``{"type": "text", ...}`` or
    ``{"type": "image_url", "image_url": {"url": ...}}``. ``system``-role text is
    collected separately (it feeds the model's system slot); ``user``-role text
    becomes the prompt. Text from multiple messages of the same role is joined.
    Returns ``(system_text, user_text, images)``.
    """
    system_parts: List[str] = []
    user_parts: List[str] = []
    images: List[str] = []
    for msg in messages:
        bucket = system_parts if msg.role == "system" else user_parts
        content = msg.content
        if isinstance(content, str):
            if content.strip():
                bucket.append(content.strip())
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    t = (part.get("text") or "").strip()
                    if t:
                        bucket.append(t)
                elif ptype == "image_url":
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if url:
                        images.append(url)
    return "\n".join(system_parts).strip(), "\n".join(user_parts).strip(), images


def create_app(engine: MobileOEngine, model_id: str = DEFAULT_MODEL_ID) -> FastAPI:
    app = FastAPI(title="Mobile-O OpenAI-compatible server")

    @app.get("/v1/models")
    def list_models():
        # Minimal OpenAI model-listing shape; the daemon uses this purely as a
        # liveness probe but real OpenAI clients also accept it.
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "mobile-o"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, request: Request):
        client = request.client.host if request.client else "?"
        system_text, text, images = _extract_messages(req.messages)

        if not images:
            # Expected client mistake (text-only / empty image_url) — reject with
            # 400 but log at DEBUG so it doesn't spam the log at WARNING level.
            return _error(400, "Mobile-O requires an image. Send an 'image_url' "
                               "content part (base64 data URI, bare base64, or URL).",
                          client=client,
                          detail=f"model={req.model!r} messages={len(req.messages)} "
                                 f"has_text={bool(text)} images=0",
                          level=logging.DEBUG)

        # Mode resolution: explicit `mode` wins; otherwise free-form text => the
        # model's `prompt` mode, and no text => the default `caption` prompt.
        if req.mode:
            mode, prompt_text = req.mode, text
        elif text:
            mode, prompt_text = "prompt", text
        else:
            mode, prompt_text = engine.default_mode, ""

        result = engine.run_understand(
            images_list=images,
            mode=mode,
            text=prompt_text,
            temperature=req.temperature,
            max_new_tokens=req.max_tokens,
            system=system_text or None,
        )

        if result.get("status") != "success":
            return _error(400, result.get("error", "inference failed"),
                          client=client,
                          detail=f"model={req.model!r} mode={mode!r} images={len(images)}")

        responses = result["data"]["responses"]
        # Mobile-O produces one response per image. A normal single-image chat
        # request yields one; if several images were sent we join them so nothing
        # is silently dropped.
        content = responses[0] if len(responses) == 1 else "\n".join(responses)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            # Token accounting isn't tracked by the engine; report zeros rather
            # than omitting the field, which some OpenAI clients require.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            # Non-standard extra: per-image responses + timings, handy for callers
            # that send multiple images or want latency breakdown.
            "mobileo": result["data"],
        }

    return app


def _error(status: int, message: str, client: str = "?", detail: str = "",
           level: int = logging.WARNING) -> JSONResponse:
    # OpenAI-style error envelope. Also log the reason so a bare "400 Bad Request"
    # in the access log has a matching explanation right next to it. Expected
    # client mistakes pass level=logging.DEBUG to stay out of the default log.
    logger.log(level, "chat/completions %d from %s: %s%s",
               status, client, message, f" [{detail}]" if detail else "")
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error"}})


def serve(engine: MobileOEngine, model_id: str = DEFAULT_MODEL_ID,
          host: str = "0.0.0.0", port: int = 8400):
    """Run the HTTP server (blocking)."""
    import copy
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    # Prepend a timestamp to uvicorn's default + access log lines (and our own
    # WARNING lines, which use the uvicorn.error logger) so every request has a
    # time next to it. Container-local time; format: 2026-07-14 12:34:56.
    log_config = copy.deepcopy(LOGGING_CONFIG)
    for name, fmt in (("default", "%(asctime)s %(levelprefix)s %(message)s"),
                      ("access", '%(asctime)s %(levelprefix)s %(client_addr)s - '
                                 '"%(request_line)s" %(status_code)s')):
        log_config["formatters"][name]["fmt"] = fmt
        log_config["formatters"][name]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    app = create_app(engine, model_id)
    print(f"OpenAI-compatible server listening on http://{host}:{port}  (model id: {model_id})")
    uvicorn.run(app, host=host, port=port, log_config=log_config)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Standalone Mobile-O OpenAI-compatible server")
    parser.add_argument("--model_path", type=str, default="checkpoints/final_merged_model_23620")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--default_mode", type=str, choices=["caption", "description"], default="caption")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_batch_images", type=int, default=8)
    cli = parser.parse_args()

    engine = MobileOEngine(
        cli.model_path,
        device=cli.device,
        default_mode=cli.default_mode,
        default_max_new_tokens=cli.max_new_tokens,
        max_batch_images=cli.max_batch_images,
    )
    serve(engine, model_id=cli.model_id, host=cli.host, port=cli.port)
