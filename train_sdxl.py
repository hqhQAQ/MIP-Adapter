import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.attention_processor_multi import IPAttnProcessor, AttnProcessor

# New added
import copy
import logging
from data.subject_dataset import SubjectDataset
from evaluation.inference_func import inference_on_concept101_dataset
from evaluation.evaluate_func import evaluate_scores


def get_state_dict(old_state_dict):
    image_proj_sd = {}
    ip_sd = {}
    for k in old_state_dict:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = old_state_dict[k].cpu().half()
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = old_state_dict[k].cpu().half()
    state_dict = {"image_proj": image_proj_sd, "ip_adapter": ip_sd}
    return state_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.stack([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    # New added
    prompts = [example["instance_prompt"] for example in data]
    tokenizer_entity_indexes = [example["tokenizer_entity_index"] for example in data]
    tokenizer_entity_ids = torch.stack([example["tokenizer_entity_ids"] for example in data])
    tokenizer_2_entity_ids = torch.stack([example["tokenizer_2_entity_ids"] for example in data])

    return {
        "images": images,
        "prompts": prompts,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "tokenizer_entity_indexes": tokenizer_entity_indexes,
        "tokenizer_entity_ids": tokenizer_entity_ids,
        "tokenizer_2_entity_ids": tokenizer_2_entity_ids,
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None, cross_attn_names=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        # New added
        self.cross_attn_names = cross_attn_names

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds, added_params):
        tokenizer_entity_indexes = added_params['tokenizer_entity_indexes']
        entity_text_embeds = added_params['entity_text_embeds']

        ip_tokens = self.image_proj_model(image_embeds)
        bsz = noisy_latents.shape[0]
        ip_tokens = ip_tokens.reshape(bsz, -1, ip_tokens.shape[-1])
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # New added
        for name in self.cross_attn_names:
            self.unet.attn_processors[name].entity_text_embeds = entity_text_embeds
            self.unet.attn_processors[name].tokenizer_entity_indexes = tokenizer_entity_indexes

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        missing_keys, unexpected_keys = self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_pkl_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--eval_data_root",
        type=str,
        default="",
        required=True,
        help="Test data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--stop_step",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # New added
    parser.add_argument("--infer_scale", type=float, default=0.75)
    parser.add_argument("--num_objects", type=int, default=2)
    parser.add_argument("--num_train_imgs", type=int, default=None)

    args = parser.parse_args()
    # New added
    args.validation_steps = args.save_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.log'))
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.info(args)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # Ip-adapter
    num_tokens = 4
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_tokens,
    )
    # Init adapter modules
    attn_procs, cross_attn_names = {}, []
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_params = {
                'num_tokens': num_tokens,
                'num_objects': args.num_objects,
                'layer_name': name,
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, attn_params=attn_params)
            missing_keys, unexpected_keys = attn_procs[name].load_state_dict(weights, strict=False)
            cross_attn_names.append(name)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path, cross_attn_names=cross_attn_names)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Optimizer
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Dataloader
    train_dataset = SubjectDataset(args.data_root, args.data_pkl_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, num_train_imgs=args.num_train_imgs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logger.info('Len dataset : {}'.format(len(train_dataset)))
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    stop_step = args.stop_step
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    num_objects, entity_size = batch["clip_images"].shape[1], batch["clip_images"].shape[-1]
                    clip_images = batch["clip_images"].reshape(bsz * num_objects, 3, entity_size, entity_size)
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype)).image_embeds
                    image_embeds = image_embeds.reshape(bsz, -1, image_embeds.shape[-1])

                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_).reshape(-1, image_embeds.shape[-1])
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # Concat

                    # Get the text embeds of entities
                    entity_ids, entity_ids_2 = batch['tokenizer_entity_ids'], batch['tokenizer_2_entity_ids']
                    batch_size, max_num_objects, seq_len = tuple(entity_ids.shape)
                    entity_ids, entity_ids_2 = entity_ids.reshape(batch_size * max_num_objects, -1), entity_ids_2.reshape(batch_size * max_num_objects, -1)
                    entity_text_embeds = text_encoder(entity_ids.to(accelerator.device), output_hidden_states=True).hidden_states[-2]
                    entity_text_embeds_2 = text_encoder_2(entity_ids_2.to(accelerator.device), output_hidden_states=True).hidden_states[-2]
                    entity_text_embeds = torch.concat([entity_text_embeds, entity_text_embeds_2], dim=-1)
                    entity_text_embeds = entity_text_embeds.reshape(batch_size, max_num_objects, seq_len, -1)

                # Add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                added_params ={
                    'entity_text_embeds': entity_text_embeds,
                    'tokenizer_entity_indexes': batch["tokenizer_entity_indexes"],
                }
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds, added_params)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    if step % 10 == 0:
                        logger.info("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                            epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            if global_step % args.validation_steps == 0 or global_step == stop_step:
                unwrapped_ip_adapter = accelerator.unwrap_model(ip_adapter, keep_fp32_wrapper=True)
                old_state_dict = unwrapped_ip_adapter.state_dict()
                state_dict = get_state_dict(old_state_dict)
                output_view_dir = os.path.join(args.output_dir, 'vis_{}'.format(global_step))
                logger.info('Begin inference...')
                inference_on_concept101_dataset(accelerator, args.pretrained_model_name_or_path, state_dict, None, args.image_encoder_path, output_view_dir, args.eval_data_root, args, scale=args.infer_scale)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    logger.info('Begin evaluating...')
                    clipt_score, clipi_score, dino_score = evaluate_scores(output_view_dir, args.eval_data_root)
                    logger.info('Step: %d, clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (global_step, clipt_score, clipi_score, dino_score))
                    save_score_dict = {
                        'clipt_score': str(clipt_score),
                        'clipi_score': str(clipi_score),
                        'dino_score': str(dino_score),
                    }
                    save_score_path = os.path.join(output_view_dir, 'all_score.json')
                    with open(save_score_path, 'w') as file:
                        json.dump(save_score_dict, file, indent=4)

            if global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "model.bin")
                # accelerator.save_state(save_path)
                unwrapped_ip_adapter = accelerator.unwrap_model(ip_adapter, keep_fp32_wrapper=True)
                old_state_dict = unwrapped_ip_adapter.state_dict()
                state_dict = get_state_dict(old_state_dict)
                torch.save(state_dict, save_path)
            
            begin = time.perf_counter()
            if global_step >= stop_step:
                break
        if global_step >= stop_step:
            break


if __name__ == "__main__":
    main()