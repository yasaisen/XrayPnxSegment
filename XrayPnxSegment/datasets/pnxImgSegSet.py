"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

from ..common.utils import load_json_data


class pnxImgSegSet(Dataset):
    def __init__(self, 
        datapath: str, 
        meta_list_path: str, 
        mask_key: str,
        skip_has_pnx: bool = False,
        split: str = None, 
        transform = None, 
        image_size = (256, 256), 
        calc_class_weights: bool = False, 
    ):
        self.datapath = datapath
        self.mask_key = mask_key
        self.skip_has_pnx = skip_has_pnx and (split == 'train')
        self.meta_list = []
        _meta_list = load_json_data(os.path.join(self.datapath, meta_list_path))
        
        if split == 'train' and calc_class_weights:
            class_counts = np.zeros(2, dtype=np.float64)
            total_pixels = 0
    
        for sample in _meta_list:
            if sample.get('split') == split:
                if self.skip_has_pnx and not sample.get('has_pnx'):
                    continue
                required_fields = ['image_path', 'mask_path']
                if all(field in sample for field in required_fields):
                    self.meta_list.append(sample)

                    if split == 'train' and calc_class_weights:
                        mask_path = os.path.join(self.datapath, sample[self.mask_key])

                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        bin_mask = (mask > 127 ).astype(np.uint8)
                        
                        background_pixels = np.sum(bin_mask == 0)
                        foreground_pixels = np.sum(bin_mask == 1)
                        class_counts[0] += background_pixels
                        class_counts[1] += foreground_pixels
                        total_pixels += mask.size

                else:
                    print(f"Warning: The sample is missing required fields, skipping this sample")

        if split == 'train' and calc_class_weights:
            class_frequencies = class_counts / total_pixels
            epsilon = 1e-8
            class_frequencies = np.maximum(class_frequencies, epsilon)
            weights = 1.0 / class_frequencies
            self.weights = weights / np.sum(weights) * len(weights)
            
            print(f"Class frequencies: {class_frequencies}")
            print(f"Class weights: {self.weights}")
        else:
            self.weights = None
                
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self
    ):
        return len(self.meta_list)
    
    def __getitem__(self, 
        idx
    ):
        img_path = os.path.join(self.datapath, self.meta_list[idx]['image_path'])
        mask_path = os.path.join(self.datapath, self.meta_list[idx][self.mask_key])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.mask_key == 'cropped_mask_path':
            coords = self.meta_list[idx]['coords']
            image = image[coords[0]:coords[1], coords[2]:coords[3]]

        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask shapes do not match: {image.shape} vs {mask.shape}")
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
    
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image.float() / 255.0
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask
    
def validate_dataset(
    dataset
):
    problematic_samples = []
    
    for i in range(len(dataset)):
        try:
            image, mask = dataset[i]
            
            unique_values = torch.unique(mask)
            
            if len(unique_values) == 1:
                problematic_samples.append({
                    'index': i,
                    'issue': f'Mask has only one value: {unique_values[0].item()}',
                    'path': dataset.meta_list[i]['mask_path']
                })
                
        except Exception as e:
            problematic_samples.append({
                'index': i,
                'issue': f'Loading error: {str(e)}',
                'path': dataset.meta_list[i]['mask_path'] if i < len(dataset.meta_list) else 'Unknown'
            })
    
    if problematic_samples:
        print(f"Found {len(problematic_samples)} problematic samples:")
        for sample in problematic_samples[:5]:
            print(f"  Index {sample['index']}: {sample['issue']}")
    
    return problematic_samples