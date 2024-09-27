import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter.ip_adapter_multi import IPAdapterXL, IPAdapterPlusXL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# New added
from accelerate import Accelerator
import json
import argparse


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


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def recover_image(img_tensor, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean = torch.FloatTensor(mean).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(mean)
    std = torch.FloatTensor(std).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(std)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()
    img_pil = Image.fromarray(img_np, 'RGB')

    return img_pil


def get_token_len(entity_name, tokenizer):
    entity_tokens = tokenizer(entity_name, return_tensors="pt").input_ids[0][1:-1]
    return len(entity_tokens)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--image_encoder_path", type=str)
    parser.add_argument("--ip_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--reference_image1_path", type=str)
    parser.add_argument("--reference_image2_path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--is_plus", type=str2bool, default=False)
    return parser.parse_args()


args = parse_args()
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "cuda"
resolution = 512
num_tokens = 4 if not args.is_plus else 16
num_objects = 2

state_dict = None
num_samples = args.num_samples
scale = args.scale
output_dir = args.output_dir

# Load model
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")

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
cur_model = IPAdapterPlusXL if args.is_plus else IPAdapterXL
if ip_ckpt is None:
    ip_model = cur_model(pipe, image_encoder_path, state_dict, ip_ckpt=None, device=device, ip_params=ip_params)
else:
    ip_model = cur_model(pipe, image_encoder_path, state_dict=None, ip_ckpt=ip_ckpt, device=device, ip_params=ip_params)
os.makedirs(output_dir, exist_ok=True)

images = [[Image.open(args.reference_image1_path)], [Image.open(args.reference_image2_path)]]
prompts = [args.prompt]
entity1_name, entity2_name = args.reference_image1_path.split('/')[-1].split('.')[0], args.reference_image2_path.split('/')[-1].split('.')[0]
entity_names = [[entity1_name, entity2_name]]
entity_indexes = [[(-1, get_token_len(entity1_name, tokenizer)), (-1, get_token_len(entity2_name, tokenizer))]]

generated_images = ip_model.generate(pil_images=images, num_samples=num_samples, num_inference_steps=30, seed=420, prompt=prompts, scale=scale, entity_names=entity_names, entity_indexes=entity_indexes)
for idx, image in enumerate(generated_images):
    image.save(os.path.join(output_dir, '{}_gen_{}.png'.format(args.prompt, idx)))