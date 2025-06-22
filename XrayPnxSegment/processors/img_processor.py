"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(
    image_size=(768, 768), 
):
    train_transform = A.Compose([
        # Geometric Transformations
        # A.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        # A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=60, p=0.4),
        A.Resize(height=image_size[0], width=image_size[1]),

        # Advanced Noise and Distortion
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.5),
            A.ISONoise(p=0.5),
        ], p=0.3),
        
        # Advanced Deformations
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
        ], p=0.2),
        
        # Cutout for Regularization
        # A.CoarseDropout(num_holes_range=(2, 8), fill='random', fill_mask=0, p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform





