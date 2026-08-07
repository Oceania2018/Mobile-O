"""Shared Mobile-O inference engine.

Loads the Mobile-O vision-language model once and exposes image-understanding
inference, so any front-end (currently the OpenAI-compatible HTTP server in
``openai_server.py``, launched by ``gpu_worker.py``) can share a single model copy
in the same process. Loading the model twice would double VRAM on the shared
RTX 4090, so anything that needs inference builds **one** ``MobileOEngine`` and
hands it to every front-end.

The model-loading, prompt pre-compilation, VRAM-bounded batching and image
decoding logic here was lifted verbatim from ``gpu_worker.py`` — behaviour is
unchanged; it has only been wrapped in a class so it can be reused.
"""

import base64
import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import httpx
import torch
from PIL import Image
from transformers import AutoTokenizer

from mobileo.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                               DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                               IMAGE_TOKEN_INDEX)
from mobileo.conversation import conv_templates
from mobileo.mm_utils import process_images, tokenizer_image_token
from mobileo.model import mobileoConfig, mobileoForInferenceLM
from mobileo.utils import disable_torch_init

# Built-in fixed-prompt modes. ``prompt`` mode supplies its own text and is not
# pre-compiled. Kept identical to the original worker so existing callers see the
# same outputs.
MODE_PROMPTS = {
    "caption":     "Caption this image in under 16 words.",
    "description": "Describe the image in under 32 words.",
}
VALID_MODES = {"caption", "description", "prompt"}


class MobileOEngine:
    """A loaded Mobile-O model plus the inference + image-decoding helpers.

    Construct once (it loads weights, pre-compiles the fixed prompts and runs a
    warmup pass), then call :meth:`run_understand` from any number of threads.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        default_mode: str = "caption",
        default_temperature: float = 0.0,
        default_max_new_tokens: int = 64,
        max_batch_images: int = 8,
        understand_only: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU.")
            device = "cpu"

        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.default_mode = default_mode
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.max_batch_images = max_batch_images

        print(f"Loading model from {model_path} (device={device}, dtype={self.dtype}) ...")
        disable_torch_init()
        warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # mobileoConfig, not AutoConfig: config.json declares model_type
        # "mobile_o_inference" while the registry entry is "mobileo_inference", so the
        # Auto* lookup raises. from_pretrained() normally sidesteps this via the
        # model's config_class — do the same here.
        config = mobileoConfig.from_pretrained(model_path)
        if understand_only:
            # LlavaMetaModel builds the Sana DiT + VAE + diffusion connector for any
            # config carrying ``diffusion_name_or_path``, and device_map="auto" parks
            # all of it in VRAM (~1.7 GiB). Those modules are reachable only from
            # generate_image()/sample_images(); the understanding path this server
            # serves calls generate() with und_images and never touches them (see
            # llava_arch.prepare_inputs_labels_for_multimodal, where the VAE is used
            # only when gen_images is not None). Dropping the attribute skips both
            # module construction and the corresponding weight loading, and avoids
            # the scheduler's from_pretrained() network call at startup.
            if hasattr(config, "diffusion_name_or_path"):
                delattr(config, "diffusion_name_or_path")

        self.model = mobileoForInferenceLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            device_map=device if device == "cpu" else "auto",
        )

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.image_processor = self.model.get_vision_tower().image_processor

        # Pre-compile the fixed-prompt input ids once; ``prompt`` mode builds its
        # ids per request from the caller's text.
        print("Pre-compiling fixed prompts ...")
        self._cached_input_ids_map = {}
        for mode, prompt in MODE_PROMPTS.items():
            self._cached_input_ids_map[mode] = self.build_input_ids(prompt)
            print(f"  [{mode}] pre-compiled: {self._cached_input_ids_map[mode].shape[1]} tokens")

        self._decode_executor = ThreadPoolExecutor(max_workers=min((os.cpu_count() or 4), 8))

        self._warmup()
        if device == "cuda":
            print(f"VRAM: {torch.cuda.memory_allocated() / 2**20:.0f} MiB weights+activations, "
                  f"{torch.cuda.memory_reserved() / 2**20:.0f} MiB reserved by the allocator")
        print("Model ready.")

    # ------------------------------------------------------------------
    # Prompt / input-id construction
    # ------------------------------------------------------------------
    def build_input_ids(self, text: str, system: Optional[str] = None):
        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen_2"].copy()
        if system:
            # Override the template's default system content ("You are a helpful
            # assistant.") while keeping the ChatML "<|im_start|>system\n" prefix
            # the qwen_2 format expects (get_prompt appends the <|im_end|> sep).
            conv.system = "<|im_start|>system\n" + system
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        return (
            tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

    def _warmup(self):
        print("Running warmup pass ...")
        warmup_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
        warmup_img = process_images([warmup_image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            self.model.generate(
                self._cached_input_ids_map["caption"],
                images=warmup_img.unsqueeze(0).to(self.dtype),
                do_sample=False,
                num_beams=1,
                max_new_tokens=8,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(self, images: List[Image.Image], temperature: float, max_new_tokens: int, input_ids) -> List[str]:
        """Chunk large image batches to bound peak VRAM, then release the CUDA cache.

        PyTorch's caching allocator never returns freed blocks to the driver, so the
        process steadily holds whatever the largest batch ever needed. Capping the
        per-batch image count bounds that peak; emptying the cache after each task
        keeps the high-water mark from creeping up over long uptimes. Outputs are
        identical to processing the full list in one batch — only the grouping changes.
        """
        cap = max(1, self.max_batch_images)
        try:
            if len(images) <= cap:
                return self._infer_batch(images, temperature, max_new_tokens, input_ids)
            results: List[str] = []
            for start in range(0, len(images), cap):
                chunk = images[start:start + cap]
                results.extend(self._infer_batch(chunk, temperature, max_new_tokens, input_ids))
            return results
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _infer_batch(self, images: List[Image.Image], temperature: float, max_new_tokens: int, input_ids) -> List[str]:
        gen_kwargs = dict(
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        image_tensors = process_images(images, self.image_processor, self.model.config)

        # Batch path: process_images returned a single [N, C, H, W] tensor
        if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
            batched_input_ids = input_ids.repeat(len(images), 1)
            with torch.inference_mode():
                output_ids = self.model.generate(batched_input_ids, images=image_tensors.to(self.dtype), **gen_kwargs)
            return [s.strip() for s in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

        # Fallback: anyres returned tensors with different shapes (different patch counts).
        # Resize all images to the processor's fixed crop size so shapes unify, then retry batch.
        target_size = self.image_processor.crop_size["height"]
        resized_images = [img.resize((target_size, target_size), Image.BICUBIC) for img in images]
        image_tensors = process_images(resized_images, self.image_processor, self.model.config)

        if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
            batched_input_ids = input_ids.repeat(len(images), 1)
            with torch.inference_mode():
                output_ids = self.model.generate(batched_input_ids, images=image_tensors.to(self.dtype), **gen_kwargs)
            return [s.strip() for s in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

        # Last resort: loop (should not normally be reached)
        results = []
        tensor_list = image_tensors if isinstance(image_tensors, list) else [image_tensors[i] for i in range(len(images))]
        for img_tensor in tensor_list:
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=img_tensor.unsqueeze(0).to(self.dtype), **gen_kwargs)
            results.append(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
        return results

    # ------------------------------------------------------------------
    # Image resolution (base64 / data URI / URL)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_url(s: str) -> bool:
        return s.lower().startswith(("http://", "https://"))

    @staticmethod
    def _decode_b64_entry(entry: str) -> bytes:
        if entry.startswith("data:"):
            _, b64data = entry.split(",", 1)
        else:
            b64data = entry
        return base64.b64decode(b64data)

    @staticmethod
    def _fetch_url_bytes(url: str) -> bytes:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
        response.raise_for_status()
        return response.content

    def _resolve_image_entry(self, entry_tuple: tuple) -> tuple:
        i, entry = entry_tuple
        entry = str(entry).strip()
        if self._is_url(entry):
            return i, self._fetch_url_bytes(entry)
        return i, self._decode_b64_entry(entry)

    @staticmethod
    def _decode_pil(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def resolve_images(self, images_list: List[str]) -> List[Image.Image]:
        """Resolve a list of base64 / data-URI / URL strings to PIL images.

        Raises ``ValueError`` with a message identifying the first bad entry.
        """
        n = len(images_list)
        resolved: dict = {}
        errors: dict = {}

        with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
            future_to_idx = {
                pool.submit(self._resolve_image_entry, (i, entry)): i
                for i, entry in enumerate(images_list)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    idx, data = future.result()
                    resolved[idx] = data
                except httpx.HTTPStatusError as exc:
                    entry = str(images_list[i]).strip()
                    errors[i] = f"Failed to fetch image URL '{entry}': HTTP {exc.response.status_code}"
                except Exception as exc:
                    errors[i] = f"Invalid image at images[{i}]: {exc}"

        if errors:
            raise ValueError(errors[min(errors)])

        decode_futures = {self._decode_executor.submit(self._decode_pil, resolved[i]): i for i in range(n)}
        pil_images: List[Optional[Image.Image]] = [None] * n
        for future in as_completed(decode_futures):
            i = decode_futures[future]
            try:
                pil_images[i] = future.result()
            except Exception as exc:
                raise ValueError(f"Invalid image bytes at images[{i}]: {exc}")
        return pil_images

    # ------------------------------------------------------------------
    # High-level task
    # ------------------------------------------------------------------
    def run_understand(
        self,
        images_list: List[str],
        mode: Optional[str] = None,
        text: str = "",
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> dict:
        """Resolve images, run inference, and return a result dict.

        Shape matches the original worker handler:
            {"status": "success", "data": {"responses": [...], "elapsed_seconds": ..,
                                            "image_decode_seconds": .., "inference_seconds": ..}}
            {"status": "error",   "error": "..."}
        """
        if not images_list:
            return {"status": "error",
                    "error": "Missing 'images'. Provide a list of base64, data URI, or URL strings."}
        if not isinstance(images_list, list):
            images_list = [images_list]

        mode = mode or self.default_mode
        if mode not in VALID_MODES:
            return {"status": "error", "error": f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}."}

        temperature = self.default_temperature if temperature is None else float(temperature)
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        # Normalize the system prompt; empty/whitespace means "use template default".
        sys_prompt = system.strip() if isinstance(system, str) and system.strip() else None

        if mode == "prompt":
            if not text:
                return {"status": "error", "error": "'text' is required when mode=prompt."}
            input_ids = self.build_input_ids(text, sys_prompt)
        elif sys_prompt:
            # Custom system with a fixed-mode prompt: the pre-compiled ids bake the
            # default system, so build fresh ids for this request.
            input_ids = self.build_input_ids(MODE_PROMPTS[mode], sys_prompt)
        else:
            input_ids = self._cached_input_ids_map[mode]

        t_img_start = time.perf_counter()
        try:
            pil_images = self.resolve_images(images_list)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        t_img_elapsed = round(time.perf_counter() - t_img_start, 3)

        t_infer_start = time.perf_counter()
        try:
            responses = self.infer(pil_images, temperature, max_new_tokens, input_ids)
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

    def shutdown(self):
        self._decode_executor.shutdown(wait=False)
