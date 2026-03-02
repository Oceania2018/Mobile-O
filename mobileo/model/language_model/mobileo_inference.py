from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2ForCausalLM, Qwen2Config, Qwen2Model
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class mobileoConfig(Qwen2Config):
    model_type = "mobileo_inference"


class mobileoModel(LlavaMetaModel, Qwen2Model):
    config_class = mobileoConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(mobileoModel, self).__init__(config)


class mobileoForInferenceLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = mobileoConfig

    def __init__(self, config):
        super(mobileoForInferenceLM, self).__init__(config)
        config.model_type = "mobileo_inference"
        config.is_train = False
        self.model = mobileoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                und_images=images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    @torch.no_grad()
    def generate_image(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        with_cfg: bool = True,
        **kwargs,
    ):  
        self.to(torch.bfloat16)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if pixel_values is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                und_images=pixel_values,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        self.model = self.model.to(torch.bfloat16)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        img_hidden_states = outputs.hidden_states
        output_img = self.sample_images(img_hidden_states, with_cfg=with_cfg)
        return output_img
    
    def sample_images(
        self,
        pred_latents,
        with_cfg: bool = False,
        guidance_scale: float = 1.5,
        num_inference_steps: int = 20,
        num_images_per_prompt: int = 1,
        return_tensor=False,
    ):
        device = pred_latents[0].device

        batch_size = pred_latents[0].shape[0]

        if with_cfg:
            pred_latents = tuple(torch.cat([torch.zeros_like(layer), layer], dim=0) for layer in pred_latents)

        encoder_hidden_states = self.model.diffusion_connector(pred_latents).float()

        latent_size = self.get_model().dit.config.sample_size
        latent_channels = self.get_model().dit.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=None, device=device, dtype=torch.float32,
        )

        self.model.noise_scheduler.set_timesteps(num_inference_steps)

        for i,t in enumerate(tqdm(self.model.noise_scheduler.timesteps, desc="Sampling images")):
            if with_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            if hasattr(self.model.noise_scheduler, "scale_model_input"):
                latent_model_input = self.model.noise_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model.dit(
                hidden_states=latent_model_input.to(torch.bfloat16),
                encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device),
                encoder_attention_mask=None
            ).sample.float()

            if with_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.model.noise_scheduler.step(noise_pred, t, latents).prev_sample
        samples = self.decode_latents(latents.to(self.model.vae.dtype), return_tensor=return_tensor)
        return samples

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.model.vae is not None:
            latents = latents / self.model.vae.config.scaling_factor
            if "shift_factor" in self.model.vae.config and self.model.vae.config.shift_factor is not None:
                latents = latents + self.model.vae.config.shift_factor
            samples = self.model.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples
    

AutoConfig.register("mobileo_inference", mobileoConfig)
AutoModelForCausalLM.register(mobileoConfig, mobileoForInferenceLM)
