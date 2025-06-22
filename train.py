"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader

from XrayPnxSegment.common.utils import (
    plot_training_comparison, 
    predict_and_visualize, 
)
from XrayPnxSegment.datasets.pnxImgSegSet import pnxImgSegSet, validate_dataset
from XrayPnxSegment.models.modeling_segModels import (
    get_DeepLabV3Plus, 
    get_Unet, 
)
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.trainer.building_SegModelTrainer import get_lossFunc, get_optim, train_model


CONFIG = {
    'bsz': 12,
    'lr': 1e-4,
    'num_epoch': 100,
    'img_size': (768, 768),
    'mask_key': 'cropped_mask_path',  # 'mask_path', 'cropped_mask_path'
    'skip_has_pnx': True,             # Skip samples with 'has_pnx' set to False
    'calc_class_weights': True,       # Calculate class weights based on dataset
    'criterion': 'combined',          # 'BCE', 'combined'
    'root_path': '/home/yasaisen/Desktop/250610', # os.getcwd(),
    'root_path': '/home/yasaisen/Desktop/250610', # os.getcwd(),
    'meta_path': 'data_2506201607.json',
    'save_path': os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M")),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def main():
    print(f'Using device: {CONFIG["device"]}')
    os.makedirs(CONFIG['save_path'], exist_ok=True)
    
    train_transform, val_transform = get_transform(
        image_size=CONFIG['img_size'],
    )

    train_dataset = pnxImgSegSet(
        datapath=CONFIG['root_path'], 
        meta_list_path=CONFIG['meta_path'], 
        mask_key=CONFIG['mask_key'],
        skip_has_pnx=CONFIG['skip_has_pnx'],
        split='train', 
        transform=train_transform, 
        image_size=CONFIG['img_size'],
        calc_class_weights=CONFIG['calc_class_weights'],
    )
    print("Checking training dataset...")
    _ = validate_dataset(train_dataset)
    print(f'Training samples: {len(train_dataset)}')

    val_dataset = pnxImgSegSet(
        datapath=CONFIG['root_path'], 
        meta_list_path=CONFIG['meta_path'], 
        mask_key=CONFIG['mask_key'],
        split='test', 
        transform=val_transform, 
        image_size=CONFIG['img_size'],
    )
    print("Checking validation dataset...")
    _ = validate_dataset(val_dataset)
    print(f'Validation samples: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['bsz'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['bsz'], 
        shuffle=False, 
        num_workers=4
    )
    

    criterion = get_lossFunc(
        lossFunc=CONFIG['criterion'],
        class_weights=train_dataset.weights
    )
    


    print("\n" + "="*60)
    print("Training DeepLabV3Plus")
    print("="*60)
    
    deeplabv3_model = get_DeepLabV3Plus(
        device=CONFIG['device'],
    )
    deeplabv3_optimizer, deeplabv3_scheduler = get_optim(
        model=deeplabv3_model,
        lr=CONFIG['lr'],
    )
    deeplabv3_history = train_model(
        model=deeplabv3_model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=deeplabv3_optimizer, 
        scheduler=deeplabv3_scheduler, 
        num_epochs=CONFIG['num_epoch'], 
        device=CONFIG['device'], 
        model_name='deeplabv3plus', 
        save_dir=CONFIG['save_path'],
    )
    


    print("\n" + "="*60)
    print("Training U-Net")
    print("="*60)
    
    unet_model = get_Unet(
        device=CONFIG['device'],
    )
    unet_optimizer, unet_scheduler = get_optim(
        model=unet_model,
        lr=CONFIG['lr'],
    )
    unet_history = train_model(
        model=unet_model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=unet_optimizer, 
        scheduler=unet_scheduler, 
        num_epochs=CONFIG['num_epoch'], 
        device=CONFIG['device'], 
        model_name='unet', 
        save_dir=CONFIG['save_path'],
    )
    
    plot_training_comparison(
        deeplabv3_history, 
        unet_history, 
        save_dir=CONFIG['save_path']
    )

if __name__ == "__main__":
    main()









