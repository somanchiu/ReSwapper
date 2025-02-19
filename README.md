# ReSwapper

The model architectures of InSwapper and SimSwap are extremely similar. This branch is based on the SimSwap repository. Work in progress.

## Installation
  ```bash
  git clone https://github.com/somanchiu/ReSwapper.git
  cd ReSwapper
  python -m venv venv

  venv\scripts\activate

  pip install onnxruntime moviepy tensorboard timm==0.5.4 insightface==0.7.3

  pip install torch torchvision --force --index-url https://download.pytorch.org/whl/cu121
  pip install onnxruntime-gpu --force --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

  pip install sympy==1.13.1
  pip install typing_extensions --upgrade
  ```

## Training
The training now works, but it's unstable. The discriminator losses fluctuate heavily.

### 0. Pretrained weights
1. Download [arcface_w600k_r50.pth](https://huggingface.co/somanchiu/reswapper/tree/main) or convert the w600k_r50.onnx to arcface_w600k_r50.pth yourself using weight_transfer.arcface_onnx_to_pth
2. (Optional) Download [\<step>_net_D.pth](https://huggingface.co/somanchiu/reswapper/tree/main/GAN) and [\<step>_net_G.pth](https://huggingface.co/somanchiu/reswapper/tree/main/GAN) and place it in the folder "checkpoints/reswapper" 

### 1. Dataset Preparation
Download [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ)

### 2. Model Training

Example:
```python
python train.py --use_tensorboard "True" --dataset "VGGface2_HQ/VGGface2_None_norm_512_true_bygfpgan" --name "reswapper" --load_pretrain "checkpoints/reswapper" --sample_freq "1000" --model_freq "1000" --batchSize "4" --lr_g "0.00005" --lr_d "0.00005" --load_optimizer "False"
```

### Notes
- batchSize must be greater than or equal to 2
- Tested args:
  - For the steps from 1 to 6500: --lr_g "0.00005" --lr_d "0.0001" --lambda_feat "1" --batchSize "4"
  - For the steps from 6501 to 40000: --lr_g "0.00005" --lr_d "0.00005" --lambda_feat "1" --batchSize "4"
  - For the steps starting from 40001: --lr_g "0.00005" --lr_d "0.00001" --lambda_feat "1" --batchSize "4"

## To do
- Improve the stability of the training process