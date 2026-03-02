# Example run
# python inference_und.py /path/to/mobileo_unified_model
import torch
from PIL import Image
from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/mobileo_unified_1.5B")
parser.add_argument("--image_path", type=str, default="assets/funny_image.jpeg")
parser.add_argument(
    "--mode",
    type=str,
    choices=["caption", "description", "prompt"],
    default="caption",
    help="caption: Caption the image | description: Describe the image | prompt: provide custom text via --text",
)
parser.add_argument("--text", type=str, default="", help="Custom prompt text, used when --mode=prompt")
args = parser.parse_args()

MODE_PROMPTS = {
    "caption": "Caption the image",
    "description": "Describe the image",
}

if args.mode == "prompt":
    if not args.text:
        parser.error("--text is required when --mode=prompt")
    user_prompt = args.text
else:
    user_prompt = MODE_PROMPTS[args.mode]

disable_torch_init()
tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(torch.bfloat16).to("cuda:0")

image_processor = model.get_vision_tower().image_processor

qs = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
conv = conv_templates["qwen_2"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

model.generation_config.pad_token_id = tokenizer.pad_token_id
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

image_tensor = process_images([Image.open(args.image_path).convert("RGB")], image_processor, model.config)[0]

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).to(torch.bfloat16),
        do_sample=True,
        temperature=0.8,
        top_p=None,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())

