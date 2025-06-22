# XrayPnxSegment
Perform image segmentation on chest X-ray to obtain the scope of pneumothorax lesions

XrayPnxSegment provides a training pipeline for segmenting pneumothorax lesions from chest X-ray images. It contains scripts for preparing the dataset, defining neural network models, and training either **DeepLabV3+** or **U-Net** architectures using PyTorch.

## Features

- Dataset preparation utilities for cropping and sampling images
- Data loaders with `albumentations` based augmentations
- Ready-made model builders for DeepLabV3+ and U-Net
- Training routine with configurable loss functions and metrics

## Requirements

The codebase relies on Python 3 with the following libraries (incomplete list):

- `torch`
- `torchvision`
- `segmentation_models_pytorch`
- `albumentations`
- `opencv-python`

Install the dependencies with `pip` before running the scripts.

## Dataset preparation

The `preparing_data.py` script processes chest X-ray images and masks. It crops the masks to squares, creates subsets for training and testing, and saves the metadata as a JSON file.

```bash
python preparing_data.py
```

Modify the constants at the top of the script to point to your dataset directory. The resulting JSON file is used by the training pipeline.

## Training

`train.py` trains both DeepLabV3+ and U-Net models. Configuration settings such as batch size, learning rate and paths are stored in the `CONFIG` dictionary.

```bash
python train.py
```

Model checkpoints and training history will be saved under `checkpoints/`.

## Inference example

After training, you can visualize predictions using the `predict_and_visualize` function in `XrayPnxSegment/common/utils.py`:

```python
from XrayPnxSegment.common.utils import predict_and_visualize
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus

model = get_DeepLabV3Plus()
model.load_state_dict(torch.load('path/to/model.pth')['model_state_dict'])
predict_and_visualize(model, 'example.png', device='cuda')
```

## Project structure

```
XrayPnxSegment/
├── common/         # Utility functions
├── datasets/       # Dataset classes and helpers
├── models/         # Model definitions
├── processors/     # Data augmentation pipeline
└── trainer/        # Training loops and loss functions
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
