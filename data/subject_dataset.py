import os
import cv2
cv2.setNumThreads(0)
import types
import random
import argparse
from pathlib import Path
import json
import itertools
from typing import Optional
from io import BytesIO
import time
import yaml
import torch
from einops import rearrange
from torchvision.transforms import functional
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import concurrent.futures
import os
import pickle
from transformers import CLIPTokenizer, T5Tokenizer

max_num_objects = 2
area_min = 0.08
area_max = 0.7
ratio_min = 0.3
ratio_max = 3
score = 0.3
iou_ratio = 0.8
fill_bbox_ratio = 0.6
max_bbox_num_subj = 5

SKS_ID = 48136

MAX_NUM_OBJECTS = 2


def replace_clip_embeddings(clip_model, image_infos):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        if image_infos["image_token_mask"] is not None:

            shape_mask = inputs_embeds[image_infos["image_token_mask"]].shape[0]
            shape_embedding = image_infos["image_embedding"].shape[0]
            assert shape_mask == shape_embedding or (
                shape_mask == 4 and shape_embedding == 1)
            inputs_embeds[image_infos["image_token_mask"]
                          ] = image_infos["image_embedding"]

        embeddings = inputs_embeds + position_embeddings
        return embeddings
    clip_model.text_model.embeddings.old_forward = clip_model.text_model.embeddings.forward
    clip_model.text_model.embeddings.forward = types.MethodType(
        forward, clip_model.text_model.embeddings
    )


def is_valid_bbox(mask, bbox):
    crop_mask = mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    colors, mask_area = crop_mask.unique(return_counts=True)
    # Limit the number of entities within a bbox
    if len(mask_area) > max_bbox_num_subj or len(mask_area) < 2:
        return False
    # The proportion of the main mask area within the box is greater than the threshold
    if mask_area.max()/(bbox[2]-bbox[0])/(bbox[3]-bbox[1]) < fill_bbox_ratio:
        return False
    # Filter out situations where the mask area of the main body is larger than that of the bbox
    if (mask == colors[mask_area[1:].argmax()+1]).sum() > (bbox[2]-bbox[0])*(bbox[3]-bbox[1]):
        return False
    return True


def is_contained(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 > x1 and y2 > y1:
        ratio = (y2-y1)*(x2-x1)/min((box1[2]-box1[0]) *
                                    (box1[3]-box1[1]), (box2[2]-box2[0])*(box2[3]-box2[1]))
        if ratio > iou_ratio:
            return True
    return False


def fill_cavity(input_mask):
    # fast but not accurate
    cumsum = input_mask.cumsum(-1)
    filled_mask = (cumsum > 0)
    filled_mask &= (cumsum < cumsum[..., -1:])
    cumsum = input_mask.cumsum(-2)
    filled_mask &= (cumsum > 0)
    filled_mask &= (cumsum < cumsum[..., -1:, :])
    return filled_mask


def post_bbox_filter(bbox):
    if (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) > 200 and ratio_min < (bbox[3]-bbox[1])/(bbox[2]-bbox[0]) < ratio_max:
        return True
    else:
        return False


def image_seg(bbox, pixel_seg):

    cropped_seg = pixel_seg[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    colors, mask_counts = torch.unique(
        cropped_seg, return_counts=True, sorted=True)
    if len(mask_counts) > 1:
        max_mask_idx = mask_counts[1:].argmax()+1
    elif len(mask_counts) == 1:
        max_mask_idx = 0
    else:
        return torch.zeros_like(pixel_seg)
    return fill_cavity((pixel_seg == colors[max_mask_idx]).float())


def get_entity_index(input_ids, tokenizer, entity):
    input_ids = input_ids[0]
    entity_ids = tokenizer(entity, return_tensors="pt").input_ids[0][1:-1]
    hits = [torch.equal(input_ids[i:i + entity_ids.size(0)], entity_ids) for i in range(len(input_ids) - len(entity_ids))]
    indices = [i for i, hit in enumerate(hits) if hit]

    if len(indices) == 0:
        return (-1, len(entity_ids))
    else:
        return (indices[0], len(entity_ids))    # len(entity_ids): the length of the entity tokens


def make_prompt(tokenizer, tokenizer_2, ori_prompt, entities):
    template_text = ori_prompt

    input_ids = tokenizer(
        template_text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    input_ids_2 = tokenizer_2(
        template_text,
        max_length=tokenizer_2.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    # Get the index of entity token ids in the prompt token ids
    tokenizer_entity_index, tokenizer_2_entity_index = [], []
    for entity in entities:
        tokenizer_entity_index.append(get_entity_index(input_ids, tokenizer, entity))
        tokenizer_2_entity_index.append(get_entity_index(input_ids_2, tokenizer_2, entity))
    # Get the token ids of entities
    tokenizer_entity_ids, tokenizer_2_entity_ids = [], []
    for entity in entities:
        entity_ids = tokenizer(
            entity,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        entity_ids_2 = tokenizer_2(
            entity,
            max_length=tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        tokenizer_entity_ids.append(entity_ids)
        tokenizer_2_entity_ids.append(entity_ids_2)
    tokenizer_entity_ids = torch.cat(tokenizer_entity_ids, dim=0)
    tokenizer_2_entity_ids = torch.cat(tokenizer_2_entity_ids, dim=0)

    return template_text, input_ids, input_ids_2, tokenizer_entity_index, tokenizer_2_entity_index, tokenizer_entity_ids, tokenizer_2_entity_ids


def recover_image_old(img_tensor, mean_old=(0.5, 0.5, 0.5), std_old=(0.5, 0.5, 0.5)):
    mean_old = torch.FloatTensor(mean_old)
    std_old = torch.FloatTensor(std_old)
    mean_old = mean_old.view(-1, 1, 1)
    std_old = std_old.view(-1, 1, 1)
    img_tensor = img_tensor * std_old + mean_old
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()
    img_pil = Image.fromarray(img_np, 'RGB')

    return img_pil


def recover_image(img_tensor, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), mean_old=(0.5, 0.5, 0.5), std_old=(0.5, 0.5, 0.5)):
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    mean_old = torch.FloatTensor(mean_old)
    std_old = torch.FloatTensor(std_old)

    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    mean_old = mean_old.view(-1, 1, 1)
    std_old = std_old.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = img_tensor * std_old + mean_old
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()
    img_pil = Image.fromarray(img_np, 'RGB')

    return img_pil


def get_entity_image(bbox, object_segmap, instance_image, pad_white=1):
    image_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            # transforms.RandomRotation(degrees=10),
                                            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                            transforms.Resize(224),
                                            transforms.RandomCrop(224), 
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    if pad_white:
        image = (instance_image * object_segmap + 1 - object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        image = (instance_image * object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if bbox[3] - bbox[1] < bbox[2] - bbox[0]:
        pad_size = int(bbox[2] - bbox[0] - bbox[3] + bbox[1]) // 2
        image = functional.pad(image, (0, pad_size, 0, pad_size), pad_white)
    elif bbox[3] - bbox[1] > bbox[2] - bbox[0]:
        pad_size = int(-bbox[2] + bbox[0] + bbox[3] - bbox[1]) // 2
        image = functional.pad(image, (pad_size, 0, pad_size, 0), pad_white)

    return image_augmentation(image)


# Subject Dataset
class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_pkl_file, tokenizer, tokenizer_2, size=512, num_train_imgs=None, is_random=False, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate

        current_dataset_ids = []
        # read dataset json (may contation multi-datasets)
        self.data = []
        with open(data_pkl_file, 'rb') as file:
            index_list = pickle.load(file)
        for index in index_list:
            self.data.append({"index": index})
        print("Load data_id completely")
        print("Begin to filter data")

        # self.data = verify_keys(self.data)

        self.image_transforms_mask = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.CenterCrop(self.size),
                transforms.ToTensor()
            ]
        )
        self.image_transforms_mask_nocrop = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.functional._interpolation_modes_from_int(0)),
                transforms.ToTensor(),
            ]
        )
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
            ]
        )
        self.image_transforms_nocrop = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.post_image_transforms = transforms.Compose(
            [transforms.Normalize([0.5], [0.5])]
        )

    def verify_keys(self, samples, mask_size=512):
        new_data = []
        samples = [samples]
        for sample in samples:
            item = sample["index"]
            # Read json
            json_path = os.path.join(self.data_root, 'jsons', item + '.json')
            with open(json_path, "r") as f:
                file_data = json.load(f)
            # Read image
            image_path = os.path.join(self.data_root, 'images', item + '.png')
            image_im = Image.open(image_path)

            ori_width_im, ori_height_im = file_data['original_size']
            original_size = torch.Tensor([ori_height_im, ori_width_im])
            """
            ori_width_im, ori_height_im = image_im.size
            original_size = torch.Tensor([ori_height_im, ori_width_im])
            image_im = transforms.Compose([
                transforms.Resize(mask_size, interpolation=transforms.InterpolationMode.BILINEAR),
            ])(image_im)
            """
            width_im, height_im = image_im.size
            image = transforms.ToTensor()(image_im)
            # Read mask
            mask_path = os.path.join(self.data_root, 'masks', item + '.png')
            mask_im = Image.open(mask_path)
            mask = transforms.ToTensor()(mask_im)

            if "final_masks" in file_data:
                cat_prompts = [name['label'] for name in file_data['final_masks'] if "box" in name]
                bbox_ori = torch.stack([torch.tensor(name['box']) for name in file_data['final_masks'] if "box" in name])
                w_h_ratio = (bbox_ori[:, 2] - bbox_ori[:, 0]) / \
                    (bbox_ori[:, 3] - bbox_ori[:, 1])
                area_ratio = (bbox_ori[:, 2] - bbox_ori[:, 0]) * \
                    (bbox_ori[:, 3] - bbox_ori[:, 1])/width_im/height_im
                logits = torch.tensor([name["logit"] for name in file_data["final_masks"] if "box" in name])
                indices = logits.argsort(descending=True)
                bbox_selects = []
                for index in indices:
                    if area_min < area_ratio[index] < area_max and ratio_min < w_h_ratio[index] < ratio_max and logits[index] > score and " " not in cat_prompts[index] and file_data["text"].find(cat_prompts[index]) != -1:
                        if not is_valid_bbox(mask, bbox_ori[index]):
                            continue
                        # Filter high iou and duplicate entities
                        flag = True
                        for bbox_select in bbox_selects:
                            if is_contained(bbox_ori[bbox_select], bbox_ori[index]):
                                flag = False
                                break
                        if flag:
                            bbox_selects.append(index)
                if len(bbox_selects) > 0:
                    ret_val = {"jpg": image_im, "png": mask_im, "json": file_data, "original_size": original_size}
                    ret_val["bbox_selects"] = bbox_selects
                    new_data.append(ret_val)
        return new_data

    def transform(self, image_ori, bbox, bbox_selects, image_seg):
        mask_imgs = []
        mask_imgs_crop = []
        w, h = image_ori.width, image_ori.height
        min_x, min_y, max_x, max_y = int(bbox[:, 0].min()), int(
            bbox[:, 1].min()), int(bbox[:, 2].max()), int(bbox[:, 3].max())
        for bbox_select in bbox_selects:
            mask_img = np.zeros((h, w))
            x_1, y_1, x_2, y_2 = bbox[bbox_select]
            polygon = np.array([[x_1, y_1], [x_2, y_1], [x_2, y_2], [
                               x_1, y_2]], np.int32) 
            mask_img = cv2.fillConvexPoly(mask_img, polygon, (1, 1, 1))
            mask_img = Image.fromarray(mask_img)
            mask_imgs.append(mask_img)

        # Using the largest box as the drop benchmark
        crop_size = min(w, h)

        x_b, x_e = max(0, max_x - crop_size), min(min_x, w - crop_size)
        y_b, y_e = max(0, max_y - crop_size), min(min_y, h - crop_size)
        if x_b <= x_e and y_b <= y_e:
            start_x = random.randint(x_b, x_e)
            start_y = random.randint(y_b, y_e)
            instance_image_crop = functional.crop(
                image_ori, start_y, start_x, crop_size, crop_size)
            instance_image_seg_crop = functional.crop(
                image_seg, start_y, start_x, crop_size, crop_size)
            image = self.image_transforms_nocrop(instance_image_crop)
            image_seg = self.image_transforms_mask_nocrop(
                instance_image_seg_crop)
            for i, mask_img in enumerate(mask_imgs):
                mask_img = functional.crop(mask_img, start_y, start_x,
                                  crop_size, crop_size)
                mask_img = self.image_transforms_mask_nocrop(mask_img)
                mask_imgs_crop.append(1 - mask_img)
        else:
            start_y, start_x = (h - self.size) // 2, (w - self.size) // 2
            for i, mask_img in enumerate(mask_imgs):
                mask_img = self.image_transforms_mask(mask_img)
                mask_imgs_crop.append(1 - mask_img)
            image = self.image_transforms(image_ori)
            image_seg = self.image_transforms_mask(image_seg)

        crop_coords_top_left = torch.tensor([start_y, start_x])
        return image, torch.cat(mask_imgs_crop), image_seg, crop_coords_top_left

    def preproc(self, result):
        new_data = []
        result = [result]
        for sample in result:
            example = {}
            instance_image = sample["jpg"]
            w, h = instance_image.size
            if "final_masks" in sample["json"]:
                bbox_ori = torch.stack([torch.tensor(name["box"])
                                    for name in sample["json"]["final_masks"] if "box" in name])
                bbox_selects = sample["bbox_selects"]
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
                example["instance_image"], example["mask"], example["instance_seg"], example["crop_coords_top_left"] = self.transform(
                    instance_image, bbox_ori, bbox_selects, sample["png"])
                example["original_size"] = sample["original_size"]
                example["instance_prompt"] = sample["json"]["text"]
                example["cat_prompts"] = [[name["label"] for name in sample["json"]
                                        ["final_masks"] if "box" in name][bbox_select] for bbox_select in bbox_selects]
            new_data.append(example)
        return new_data

    def post_verify(self, samples, tokenizer, tokenizer_2, sample_idx, hr_size=-1):
        new_data = []
        for sample in samples:
            masks = sample["mask"]
            kept_masks = []
            kept_entities = []
            bboxes = torch.zeros((max_num_objects, 4))
            clip_image_mask = torch.zeros((max_num_objects), dtype=bool)
            padded_object_segmaps = torch.zeros((max_num_objects, hr_size, hr_size))
            entity_images = torch.zeros(max_num_objects, 3, 224, 224)
            for i, mask in enumerate(masks):
                y, x = torch.where(mask == 0)

                if len(y) == 0 or not post_bbox_filter([x.min(), y.min(), x.max(), y.max()]):
                    continue
                bbox = (x.min(), y.min(), x.max(), y.max())
                
                padded_object_segmaps[len(kept_masks)] = image_seg(bbox, sample["instance_seg"])
                entity_images[len(kept_masks)] = get_entity_image(bbox, padded_object_segmaps[len(kept_masks)], sample["instance_image"])
                clip_image_mask[len(kept_masks)] = True
                bboxes[len(kept_masks)] = torch.tensor([bbox])/512
                kept_masks.append(mask.unsqueeze(0))
                kept_entities.append(sample["cat_prompts"][i])

                if len(kept_masks) == max_num_objects:
                    break
            if len(kept_masks) <= 1:    # Filter samples without multiple objects
                continue
            elif kept_entities[0] == kept_entities[1]:  # Filter samples with equal entity names
                continue
            else:
                sample["image"] = self.post_image_transforms(sample["instance_image"])
                sample["mask"] = torch.cat(kept_masks)
                sample["cat_prompts"] = kept_entities
                sample["bbox"] = bboxes
                sample["object_segmap"] = padded_object_segmaps

                # CFG training
                instance_prompt = sample["instance_prompt"]
                drop_image_embed = 0
                rand_num = random.random()
                if rand_num < self.i_drop_rate:
                    drop_image_embed = 1
                elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                    instance_prompt = ""
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                    instance_prompt = ""
                    drop_image_embed = 1

                template_text, input_ids, input_ids_2, tokenizer_entity_index, tokenizer_2_entity_index, tokenizer_entity_ids, tokenizer_2_entity_ids = make_prompt(tokenizer, tokenizer_2, instance_prompt, kept_entities)
                sample["instance_prompt"] = template_text
                sample["text_input_ids"] = input_ids
                sample["text_input_ids_2"] = input_ids_2
                sample["clip_image"] = entity_images
                sample["target_size"] = torch.tensor([self.size, self.size])
                sample["drop_image_embed"] = drop_image_embed
                sample["tokenizer_entity_index"] = tokenizer_entity_index
                sample["tokenizer_2_entity_index"] = tokenizer_2_entity_index
                sample["tokenizer_entity_ids"] = tokenizer_entity_ids
                sample["tokenizer_2_entity_ids"] = tokenizer_2_entity_ids

                new_data.append(sample)
        
        return new_data

    def getitem_info(self, idx):
        try:
            result = self.data[idx]
            result = self.verify_keys(result)[0]
            result = self.preproc(result)
            result = self.post_verify(result, self.tokenizer, self.tokenizer_2, sample_idx=idx, hr_size=self.size)[0]
            return result
        except:
            return None

    def __getitem__(self, idx): 
        while True:
            result = self.getitem_info(idx)
            if result is not None:
                return result
            else:
                idx = random.randint(0, len(self.data) -1)
                # print("WARNING:use a random idx:{} ".format(idx))
                continue

    def __len__(self):
        return len(self.data)