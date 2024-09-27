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
import os
import pickle


def resize_padding_image(img, size=224):
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1.0:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    padding_left = (size - new_width) // 2
    padding_right = size - new_width - padding_left
    padding_top = (size - new_height) // 2
    padding_bottom = size - new_height - padding_top

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=255),
    ])
    transformed_img = transform(img)
    return transformed_img


def resize_image(img, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    transformed_img = transform(img)
    return transformed_img


# Subject Dataset
class SubjectMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, tokenizer, tokenizer_2, size=512, process_size=800, num_train_imgs=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size

        # read dataset json (may contation multi-datasets)
        self.process = transforms.Compose([
            transforms.Resize(process_size)])
        self.img_dir = os.path.join(data_root, 'benchmark_dataset')
        self.mask_dir = os.path.join(data_root, 'mask_imgs')
        self.json_dir = os.path.join(data_root, 'json_data')
        self.name_to_paths = {}
        self.data = []
        anno_file = os.path.join(data_root, 'dataset_multiconcept.json')
        with open(anno_file, 'r') as f:
            samples = json.load(f)
        for comb_idx, sample in enumerate(samples):
            entity1_dict, entity2_dict = sample[0], sample[1]
            prompt_path = os.path.join(data_root, sample[2]['prompt_filename_compose'])
            entity1_name, entity2_name = entity1_dict['class_prompt'], entity2_dict['class_prompt']
            # Entity 1
            entity1_concept_name = entity1_dict['instance_data_dir'].split('/')[-1]
            entity1_img_paths = os.listdir(os.path.join(data_root, 'benchmark_dataset', entity1_concept_name))
            entity1_img_names = [x.split('.')[0] for x in entity1_img_paths]
            entity1_name_to_path = {}
            for img_name, img_path in zip(entity1_img_names, entity1_img_paths):
                entity1_name_to_path[img_name] = img_path
            if entity1_concept_name not in self.name_to_paths:
                self.name_to_paths[entity1_concept_name] = entity1_name_to_path
            # Entity 2
            entity2_concept_name = entity2_dict['instance_data_dir'].split('/')[-1]
            entity2_img_paths = os.listdir(os.path.join(data_root, 'benchmark_dataset', entity2_concept_name))
            entity2_img_names = [x.split('.')[0] for x in entity2_img_paths]
            entity2_name_to_path = {}
            for img_name, img_path in zip(entity2_img_names, entity2_img_paths):
                entity2_name_to_path[img_name] = img_path
            if entity2_concept_name not in self.name_to_paths:
                self.name_to_paths[entity2_concept_name] = entity2_name_to_path

            with open(prompt_path, 'r') as file:
                lines = file.readlines()
            for prompt in lines:
                prompt = prompt.replace('\n', '')
                prompt = prompt.replace('{0}', entity1_name)
                prompt = prompt.replace('{1}', entity2_name)
                data_dict = {
                    'comb_idx': comb_idx,
                    'index': len(self.data),
                    'prompt': prompt,
                    'entity1_concept_name': entity1_concept_name,
                    'entity1_img_names': entity1_img_names,
                    'entity1_name': entity1_name,
                    'entity2_concept_name': entity2_concept_name,
                    'entity2_img_names': entity2_img_names,
                    'entity2_name': entity2_name,
                }
                self.data.append(data_dict)

    def get_entity_index(self, input_ids, tokenizer, entity):
        input_ids = input_ids[0]
        entity_ids = tokenizer(entity, return_tensors="pt").input_ids[0][1:-1]
        hits = [torch.equal(input_ids[i:i + entity_ids.size(0)], entity_ids) for i in range(len(input_ids) - len(entity_ids))]
        indices = [i for i, hit in enumerate(hits) if hit]

        if len(indices) == 0:
            return 1, 1
        else:
            return indices[0], len(entity_ids)

    def verify_keys(self, sample, mask_size=512):
        comb_idx = sample["comb_idx"]
        item = sample["index"]
        prompt = sample["prompt"]
        entity1_concept_name = sample["entity1_concept_name"]
        entity1_img_names = sample["entity1_img_names"]
        entity1_name = sample["entity1_name"]
        entity2_concept_name = sample["entity2_concept_name"]
        entity2_img_names = sample["entity2_img_names"]
        entity2_name = sample["entity2_name"]

        entity1_img_paths = [os.path.join(self.img_dir, entity1_concept_name, self.name_to_paths[entity1_concept_name][img_name]) for img_name in entity1_img_names]
        entity2_img_paths = [os.path.join(self.img_dir, entity2_concept_name, self.name_to_paths[entity2_concept_name][img_name]) for img_name in entity2_img_names]
        entity1_json_paths = [os.path.join(self.json_dir, entity1_concept_name, img_name + '.json') for img_name in entity1_img_names]
        entity2_json_paths = [os.path.join(self.json_dir, entity2_concept_name, img_name + '.json') for img_name in entity2_img_names]

        # Read image
        entity1_imgs = [self.process(Image.open(img_path)) for img_path in entity1_img_paths]
        entity2_imgs = [self.process(Image.open(img_path)) for img_path in entity2_img_paths]
        # Read json
        entity1_datas, entity2_datas = [], []
        for json_path in entity1_json_paths:
            with open(json_path, 'r') as f:
                entity1_datas.append(json.load(f))
        entity1_bboxes = [np.array(data['mask'][1]['box']).astype(np.int32) for data in entity1_datas]
        for json_path in entity2_json_paths:
            with open(json_path, 'r') as f:
                entity2_datas.append(json.load(f))
        entity2_bboxes = [np.array(data['mask'][1]['box']).astype(np.int32) for data in entity2_datas]

        use_crop = True
        if use_crop:
            processed_entity1_imgs, processed_entity2_imgs = [], []
            for entity1_bbox, entity1_img in zip(entity1_bboxes, entity1_imgs):
                entity1_img = entity1_img.crop((entity1_bbox[0], entity1_bbox[1], entity1_bbox[2], entity1_bbox[3]))
                entity1_img = resize_padding_image(entity1_img, size=224)
                processed_entity1_imgs.append(entity1_img)
            for entity2_bbox, entity2_img in zip(entity2_bboxes, entity2_imgs):
                entity2_img = entity2_img.crop((entity2_bbox[0], entity2_bbox[1], entity2_bbox[2], entity2_bbox[3]))
                entity2_img = resize_padding_image(entity2_img, size=224)
                processed_entity2_imgs.append(entity2_img)
        else:
            processed_entity1_imgs, processed_entity2_imgs = [], []
            for entity1_bbox, entity1_img in zip(entity1_bboxes, entity1_imgs):
                entity1_img = resize_image(entity1_img, size=224)
                processed_entity1_imgs.append(entity1_img)
            for entity2_bbox, entity2_img in zip(entity2_bboxes, entity2_imgs):
                entity2_img = resize_image(entity2_img, size=224)
                processed_entity2_imgs.append(entity2_img)

        # Get the entity token index in the original prompt tokens
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        prompt_token_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0][1:-1])
        entity1_index, entity1_token_len = self.get_entity_index(prompt_ids, self.tokenizer, entity1_name)
        entity2_index, entity2_token_len = self.get_entity_index(prompt_ids, self.tokenizer, entity2_name)

        ret_val = {"entity_names": [entity1_name, entity2_name],
                   "entity_imgs": [processed_entity1_imgs, processed_entity2_imgs],
                   "entity_indexes": [(entity1_index, entity1_token_len), (entity2_index, entity2_token_len)],
                   "prompt": prompt,
                   "prompt_token_len": prompt_token_len,
                   "comb_idx": comb_idx,}
        return ret_val

    def getitem_info(self, idx):
        # try:
        result = self.data[idx]
        result = self.verify_keys(result)
        return result
        # except:
        #     return None

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