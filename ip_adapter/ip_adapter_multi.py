import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

from .attention_processor_multi import AttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, state_dict, ip_ckpt=None, device="cuda", ip_params=None):
        # num_tokens=4, num_objects=2
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.state_dict = state_dict
        self.ip_ckpt = ip_ckpt
        self.num_tokens = ip_params['num_tokens']
        self.num_objects = ip_params['num_objects']

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # Load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()
        # Image proj model
        self.image_proj_model = self.init_proj()
        
        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_params = {
                    'num_tokens': self.num_tokens,
                    'num_objects': self.num_objects,
                    'layer_name': name,
                }
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    attn_params=attn_params,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if self.state_dict is None:
            if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        else:
            state_dict = self.state_dict
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        # ip_layers.load_state_dict(state_dict["ip_adapter"])
        missing_keys, unexpected_keys = ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_multi(self, pil_image=None, clip_image_embeds=None, num_prompts=None, num_objects=2):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros(num_prompts * num_objects, clip_image_embeds.shape[-1]).type(torch.float16).cuda())
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_images,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        entity_names=None,
        entity_indexes=None,
        **kwargs,
    ):
        self.set_scale(scale)

        # num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        num_prompts = len(prompt)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        img_num_per_object = [len(x) for x in pil_images]
        pil_images = [img for img_list in pil_images for img in img_list]
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_multi(pil_images, num_prompts=num_prompts)
        dim = image_prompt_embeds.shape[-1]
        # image_prompt_embeds = image_prompt_embeds.reshape(num_prompts, -1, dim)
        image_prompt_embeds = torch.split(image_prompt_embeds, img_num_per_object, dim=0)
        averaged_image_prompt_embeds = torch.stack([x.mean(dim=0) for x in image_prompt_embeds], dim=0) # (num_prompts * 2, dim)
        image_prompt_embeds = averaged_image_prompt_embeds.reshape(num_prompts, -1, dim)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape(num_prompts, -1, dim)

        bs_embed, seq_len, _ = image_prompt_embeds.shape    # bs_embed = num_prompts
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # Get the text embeds of entities (New added)
            if entity_names is not None:
                entity_names = [item for sublist in entity_names for item in sublist]
                negative_entity_names = ["" for _ in range(len(entity_names))]
                entity_text_embeds, negative_entity_text_embeds, _, _ = self.pipe.encode_prompt(
                    entity_names,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_entity_names,
                )   # (batch_size * 2 * num_samples, 77, dim)
                seq_len, dim = entity_text_embeds.shape[-2], entity_text_embeds.shape[-1]
                entity_text_embeds = entity_text_embeds.reshape(bs_embed, self.num_objects, num_samples, seq_len, dim)
                entity_text_embeds = entity_text_embeds.permute(0, 2, 1, 3, 4).reshape(bs_embed * num_samples, self.num_objects, seq_len, dim)
                negative_entity_text_embeds = negative_entity_text_embeds.reshape(bs_embed * num_samples, self.num_objects, seq_len, dim)
                entity_text_embeds = torch.cat([negative_entity_text_embeds, entity_text_embeds], dim=0)

        self.generator = get_generator(seed, self.device)

        # New added
        cross_attn_names = [name for name in self.pipe.unet.attn_processors.keys() if not name.endswith("attn1.processor")]
        if entity_names is not None:
            for idx, name in enumerate(cross_attn_names):
                self.pipe.unet.attn_processors[name].entity_text_embeds = entity_text_embeds
        if entity_indexes is not None:
            entity_indexes = [element for item in entity_indexes for element in [item for _ in range(num_samples)]]
            entity_indexes = entity_indexes + entity_indexes    # Negative + positive
            for idx, name in enumerate(cross_attn_names):
                self.pipe.unet.attn_processors[name].tokenizer_entity_indexes = entity_indexes
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds_multi(self, pil_image=None, clip_image_embeds=None, num_prompts=None, num_objects=2):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # uncond_clip_image_embeds = self.image_encoder(
        #     torch.zeros_like(clip_image), output_hidden_states=True
        # ).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros(num_prompts * num_objects, clip_image.shape[-3], clip_image.shape[-2], clip_image.shape[-1]).type(torch.float16).cuda(), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_images=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        entity_names=None,
        entity_indexes=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = len(prompt)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        img_num_per_object = [len(x) for x in pil_images]
        pil_images = [img for img_list in pil_images for img in img_list]
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_multi(
            pil_image=pil_images, clip_image_embeds=clip_image_embeds, num_prompts=num_prompts
        )
        dim = image_prompt_embeds.shape[-1]
        image_prompt_embeds = torch.split(image_prompt_embeds, img_num_per_object, dim=0)
        averaged_image_prompt_embeds = torch.stack([x.mean(dim=0) for x in image_prompt_embeds], dim=0) # (num_prompts * 2, dim)
        image_prompt_embeds = averaged_image_prompt_embeds.reshape(num_prompts, -1, dim)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.reshape(num_prompts, -1, dim)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # Get the text embeds of entities (New added)
            if entity_names is not None:
                entity_names = [item for sublist in entity_names for item in sublist]
                negative_entity_names = ["" for _ in range(len(entity_names))]
                entity_text_embeds, negative_entity_text_embeds, _, _ = self.pipe.encode_prompt(
                    entity_names,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_entity_names,
                )   # (batch_size * 2 * num_samples, 77, dim)
                seq_len, dim = entity_text_embeds.shape[-2], entity_text_embeds.shape[-1]
                entity_text_embeds = entity_text_embeds.reshape(bs_embed, self.num_objects, num_samples, seq_len, dim)
                entity_text_embeds = entity_text_embeds.permute(0, 2, 1, 3, 4).reshape(bs_embed * num_samples, self.num_objects, seq_len, dim)
                negative_entity_text_embeds = negative_entity_text_embeds.reshape(bs_embed * num_samples, self.num_objects, seq_len, dim)
                entity_text_embeds = torch.cat([negative_entity_text_embeds, entity_text_embeds], dim=0)

        self.generator = get_generator(seed, self.device)

        # New added
        cross_attn_names = [name for name in self.pipe.unet.attn_processors.keys() if not name.endswith("attn1.processor")]
        if entity_names is not None:
            for idx, name in enumerate(cross_attn_names):
                self.pipe.unet.attn_processors[name].entity_text_embeds = entity_text_embeds
        if entity_indexes is not None:
            entity_indexes = [element for item in entity_indexes for element in [item for _ in range(num_samples)]]
            entity_indexes = entity_indexes + entity_indexes    # Negative + positive
            for idx, name in enumerate(cross_attn_names):
                self.pipe.unet.attn_processors[name].tokenizer_entity_indexes = entity_indexes
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images
