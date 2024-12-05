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
Experimental starting point:

Step 1: Download [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

Step 2: Donwload the ReSwapper pretrained weight and place it in the folder "checkpoints/reswapper_512". Rename the pretrained weight to "0_net_G.pth"

Step 3:  Download the zip file from https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip, extract "people/latest_net_D1.pth" and place it in the folder "checkpoints/reswapper_512" and rename it to "0_net_D.pth"

Step 3:

```python
python train.py --use_tensorboard "True" --dataset "FFHQ" --name "reswapper_512" --load_pretrain "checkpoints/reswapper_512" --checkpoints_dir "checkpoints" --sample_freq "1000" --model_freq "1000" --lr_g "0.00005" --lr_d "0.0001" --continue_train "True" --which_epoch "0" --load_optimizer "False"
```

## To do
- Fix runtime errors: In progress
- Changing the model architecture from SimSwap to InSwapper: Done
- Experiment with the training: In progress
- Clean the code: Not started