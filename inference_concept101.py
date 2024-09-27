import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
# New added
from accelerate import Accelerator
from evaluation.inference_func import inference_on_concept101_dataset
from evaluation.evaluate_func import evaluate_scores
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
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--image_encoder_path", type=str)
    parser.add_argument("--ip_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--is_plus", type=str2bool, default=False)
    return parser.parse_args()


args = parse_args()
data_root = args.data_root
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "cuda"
resolution = 512
batch_size = 4
num_tokens = 4 if not args.is_plus else 16
num_objects = 2

state_dict = None
scale = args.scale
output_dir = args.output_dir

inference_on_concept101_dataset(accelerator, base_model_path, state_dict, ip_ckpt, image_encoder_path, output_dir, data_root, args, resolution=resolution, batch_size=batch_size, num_objects=num_objects, num_tokens=num_tokens, scale=scale, is_plus=args.is_plus)

# Evaluate scores
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    clipt_score, clipi_score, dino_score = evaluate_scores(output_dir, data_root)
    print('clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (clipt_score, clipi_score, dino_score))
    save_score_dict = {
        'clipt_score': str(clipt_score),
        'clipi_score': str(clipi_score),
        'dino_score': str(dino_score),
    }
    save_score_path = os.path.join(output_dir, 'all_score.json')
    with open(save_score_path, 'w') as file:
        json.dump(save_score_dict, file, indent=4)