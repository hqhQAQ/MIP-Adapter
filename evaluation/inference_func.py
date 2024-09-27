import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter.ip_adapter_multi import IPAdapterXL, IPAdapterPlusXL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# New added
from data.concept101_dataset import SubjectMultiDataset
from accelerate import Accelerator
import json


def collate_fn(data):
    # New added
    comb_idxes = [example["comb_idx"] for example in data]
    prompts = [example["prompt"] for example in data]
    prompt_token_lens = [example["prompt_token_len"] for example in data]
    entity_names = [example["entity_names"] for example in data]
    clip_images, entity_indexes = [], []
    for example in data:
        clip_images.extend([example["entity_imgs"][0], example["entity_imgs"][1]])
        entity_indexes.append(example["entity_indexes"])

    return {
        "comb_idxes": comb_idxes,
        "clip_images": clip_images,
        "prompts": prompts,
        "prompt_token_lens": prompt_token_lens,
        "entity_names": entity_names,
        "entity_indexes": entity_indexes,
    }


def inference_on_concept101_dataset(accelerator, base_model_path, state_dict, ip_ckpt,
                                    image_encoder_path, output_dir, data_root, args,
                                    device="cuda", resolution=512, batch_size=4,
                                    num_objects=2, num_tokens=4, scale=0.6, is_plus=False):
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")

    # Load dataset
    dataset = SubjectMultiDataset(data_root, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    # Load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_slicing()
    pipe.to(device)

    # Load ip-adapter
    ip_params = {
        'num_tokens': num_tokens,
        'num_objects': num_objects,
    }
    cur_model = IPAdapterPlusXL if is_plus else IPAdapterXL
    if ip_ckpt is None:
        ip_model = cur_model(pipe, image_encoder_path, state_dict, ip_ckpt=None, device=device, ip_params=ip_params)
    else:
        ip_model = cur_model(pipe, image_encoder_path, state_dict=None, ip_ckpt=ip_ckpt, device=device, ip_params=ip_params)
    os.makedirs(output_dir, exist_ok=True)

    for step, batch in enumerate(dataloader):
        clip_images = batch['clip_images']
        comb_idxes = batch['comb_idxes']
        prompts = batch['prompts']
        entity_names = batch['entity_names']
        entity_indexes = batch['entity_indexes']

        images = clip_images
        num_samples = 1
        generated_images = ip_model.generate(pil_images=images, num_samples=num_samples, num_inference_steps=30, seed=420, prompt=prompts, scale=scale, entity_names=entity_names, entity_indexes=entity_indexes)
        for cur_idx in range(batch_size):
            cur_prompt = prompts[cur_idx]
            comb_idx = comb_idxes[cur_idx]
            comb_dir = os.path.join(output_dir, 'comb_{}'.format(comb_idx))
            os.makedirs(comb_dir, exist_ok=True)

            cur_entity_images = images[cur_idx * num_objects] + images[cur_idx * num_objects + 1]
            # cur_entity_images[0].save(os.path.join(comb_dir, '{}_entity_{}.png'.format(cur_prompt, 0)))
            # cur_entity_images[-1].save(os.path.join(comb_dir, '{}_entity_{}.png'.format(cur_prompt, 1)))
            
            cur_generated_images = generated_images[cur_idx * num_samples: (cur_idx + 1) * num_samples]
            for idx, image in enumerate(cur_generated_images):
                image.save(os.path.join(comb_dir, '{}_gen.png'.format(cur_prompt)))
    
    # Save prompt for each generated image
    comb_dirs = [os.path.join(output_dir, x) for x in os.listdir(output_dir) if not x.endswith('.json')]
    for comb_dir in comb_dirs:
        img_names = [x for x in os.listdir(comb_dir) if 'gen' in x]
        img_prompts = [x.split('.')[0] + '.' for x in img_names]
        save_dict = {}
        for img_name, img_prompt in zip(img_names, img_prompts):
            save_dict[img_name] = img_prompt
        save_dict_path = os.path.join(comb_dir, 'prompts.json')
        with open(save_dict_path, 'w') as f:
            json.dump(save_dict, f, indent=4)