# ReSwapper

ReSwapper aims to reproduce the implementation of inswapper. This repository provides code for training, inference, and includes pretrained weights.

Here is the comparesion of the output of Inswapper and Reswapper.
| Target | Source | Inswapper Output | Reswapper Output<br>(Step 429500) |
|--------|--------|--------|--------|
| ![image](example/1/target.jpg) |![image](example/1/source.jpg) | ![image](example/1/inswapperOutput.jpg) | ![image](example/1/reswapperOutput.jpg) |
| ![image](example/2/target.jpg) |![image](example/2/source.jpg) | ![image](example/2/inswapperOutput.jpg) | ![image](example/2/reswapperOutput.jpg) |
| ![image](example/3/target.jpg) |![image](example/3/source.png) | ![image](example/3/inswapperOutput.jpg) | ![image](example/3/reswapperOutput.jpg) |

## Installation

```bash
git clone https://github.com/somanchiu/ReSwapper.git
cd ReSwapper
python -m venv venv

venv\scripts\activate

pip install -r requirements.txt

pip install torch torchvision --force --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu --force --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## The details of inswapper

### Model architecture
The inswapper model architecture can be visualized in [Netron](https://netron.app). You can compare with ReSwapper implementation to see architectural similarities

We can also use the following Python code to get more details:
```python
model = onnx.load('test.onnx')
printable_graph=onnx.helper.printable_graph(model.graph)
```

### Model inputs
- target: [1, 3, 128, 128] shape, normalized to [-1, 1] range
- source (latent): [1, 512] shape, the features of the source face
    - Calculation of latent, "emap" can be extracted from the original inswapper model.
        ```python
        latent = source_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, emap)
        latent /= np.linalg.norm(latent)
        ```

### Model output
Model inswapper_128 not only changes facial features, but also body shape.

| Target | Source | Inswapper Output | Reswapper Output<br>(Step 429500) |
|--------|--------|--------|--------|
| ![image](example/1/target.jpg) |![image](example/1/source.jpg) | ![image](example/1/inswapperOutput.gif) | ![image](example/1/reswapperOutput.gif) |

### Loss Functions
There is no information released from insightface. It is an important part of the training. However, there are a lot of articles and papers that can be referenced. By reading a substantial number of articles and papers on face swapping, ID fidelity, and style transfer, you'll frequently encounter the following keywords:
- content loss
- style loss/id loss
- perceptual loss

## Training
### 0. Pretrained weights (Optional)
If you don't want to train the model from scratch, you can download the pretrained weights and pass model_path into the train function in train.py.

### 1. Dataset Preparation
Download [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) to use as target and source images. For the swaped face images, we can use the inswapper output.

### 2. Model Training

Optimizer: Adam

Learning rate: 0.0001

Modify the code in train.py if needed. Then, execute:
```python
python train.py
```

The model will be saved as "reswapper-\<total steps\>.pth". You can also save the model as ONNX using the ModelFormat.save_as_onnx_model function. The ONNX model can then be used with the original INSwapper class.

## Notes
- Do not stop the training too early.

- I'm using an RTX3060 12GB for training. It takes around 12 hours for 50,000 steps.
- The optimizer may need to be changed to SGD for the final training, as many articles show that SGD can result in lower loss.
- To get inspiration for improving the model, you might want to review the commented code and unused functions in commit [c2a12e10021ecd1342b9ba50570a16b18f9634b9](https://github.com/somanchiu/ReSwapper/commit/c2a12e10021ecd1342b9ba50570a16b18f9634b9).

## Inference
```python
python swap.py
```

## Pretrained Model

- [reswapper-429500.pth](https://huggingface.co/somanchiu/reswapper/tree/main)
- [reswapper-429500.onnx](https://huggingface.co/somanchiu/reswapper/tree/main)

## To Do
- Create a 512-resolution model (alternative to inswapper_512)
- Implement face paste-back functionality