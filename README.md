# ReSwapper

The model architectures of InSwapper and SimSwap are extremely similar. This branch is based on the SimSwap repository. Work in progress.

## Installation
- Step 1
  ```bash
  git clone https://github.com/somanchiu/ReSwapper.git
  cd ReSwapper
  python -m venv venv

  venv\scripts\activate

  pip install torch torchvision --force --index-url https://download.pytorch.org/whl/cu121
  pip install onnxruntime-gpu --force --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

  pip install sympy==1.13.1 insightface==0.2.1 onnxruntime moviepy tensorboard timm==0.5.4
  ```

- Step 2:

  Download [arcface_checkpoint.tar](https://drive.google.com/file/d/1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N/view?usp=drive_link) and place it in the folder "arcface_model/"

## Training
Example: 
```python
python train.py
```

## To do
- Fix runtime errors: Done
- Changing the model architecture from SimSwap to InSwapper: Done
- Experiment with the training: In progress
- Clean the code: Not started