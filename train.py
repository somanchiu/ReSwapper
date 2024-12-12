from datetime import datetime
import os
import random
import torch
import torch.optim as optim

import Image
import ModelFormat
from StyleTransferLoss import StyleTransferLoss
import onnxruntime as rt

import cv2
from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis
import face_align

from StyleTransferModel_128 import StyleTransferModel
from torch.utils.tensorboard import SummaryWriter

inswapper_128_path = 'inswapper_128.onnx'
img_size = 128

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

inswapperInferenceSession = rt.InferenceSession(inswapper_128_path, providers=providers)

faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(640, 640))

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_loss_fn = StyleTransferLoss().to(get_device())

def train(datasetDir, learning_rate=0.0001, model_path=None, outputModelFolder='', saveModelEachSteps = 1, stopAtSteps=None, logDir=None, previewDir=None, saveAs_onnx = False):
    device = get_device()
    print(f"Using device: {device}")

    model = StyleTransferModel().to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded model from {model_path}")

        lastSteps = int(model_path.split('-')[-1].split('.')[0])
        print(f"Resuming training from step {lastSteps}")
    else:
        lastSteps = 0

    model.train()
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    if logDir is not None:
        train_writer = SummaryWriter(os.path.join(logDir, "training"))
        val_writer = SummaryWriter(os.path.join(logDir, "validation"))

    steps = 0

    image = os.listdir(datasetDir)

    # Training loop
    while True:
        start_time = datetime.now()

        targetFaceIndex = random.randint(0, len(image)-1)
        sourceFaceIndex = random.randint(0, len(image)-1)

        target_img=cv2.imread(f"{datasetDir}/{image[targetFaceIndex]}")
        faces = faceAnalysis.get(target_img)

        if targetFaceIndex != sourceFaceIndex:
            source_img = cv2.imread(f"{datasetDir}/{image[sourceFaceIndex]}")
            faces2 = faceAnalysis.get(source_img)
        else:
            faces2 = faces

        if len(faces) > 0 and len(faces2) > 0:
            new_aligned_face, _ = face_align.norm_crop2(target_img, faces[0].kps, img_size)
            blob = Image.getBlob(new_aligned_face)
            latent = Image.getLatent(faces2[0])
        else:
            continue

        if targetFaceIndex != sourceFaceIndex:
            input = {inswapperInferenceSession.get_inputs()[0].name: blob,
                    inswapperInferenceSession.get_inputs()[1].name: latent}

            expected_output = inswapperInferenceSession.run([inswapperInferenceSession.get_outputs()[0].name], input)[0]
        else:
            expected_output = blob

        latent_tensor = torch.from_numpy(latent).to(device)
        target_input_tensor = torch.from_numpy(blob).to(device)

        optimizer.zero_grad()
        output = model(target_input_tensor, latent_tensor)

        expected_output_tensor = torch.from_numpy(expected_output).to(device)

        content_loss, identity_loss = style_loss_fn(output, expected_output_tensor)

        loss = content_loss

        if identity_loss is not None:
            loss +=identity_loss
        
        loss.backward()

        optimizer.step()

        steps += 1
        totalSteps = steps + lastSteps

        if logDir is not None:
            train_writer.add_scalar("Loss/total", loss.item(), totalSteps)
            train_writer.add_scalar("Loss/content_loss", content_loss.item(), totalSteps)

            if identity_loss is not None:
                train_writer.add_scalar("Loss/identity_loss", identity_loss.item(), totalSteps)

        elapsed_time = datetime.now() - start_time

        print(f"Total Steps: {totalSteps}, Step: {steps}, Loss: {loss.item():.4f}, Elapsed time: {elapsed_time}")

        if steps % saveModelEachSteps == 0:
            outputModelPath = f"reswapper-{totalSteps}.pth"
            if outputModelFolder != '':
                outputModelPath = f"{outputModelFolder}/{outputModelPath}"
            saveModel(model, outputModelPath)

            validation_total_loss, validation_content_loss, validation_identity_loss, swapped_face = validate(outputModelPath)
            if previewDir is not None:
                cv2.imwrite(f"{previewDir}/{totalSteps}.jpg", swapped_face)

            if logDir is not None:
                val_writer.add_scalar("Loss/total", validation_total_loss.item(), totalSteps)
                val_writer.add_scalar("Loss/content_loss", validation_content_loss.item(), totalSteps)
                if validation_identity_loss is not None:
                    val_writer.add_scalar("Loss/identity_loss", validation_identity_loss.item(), totalSteps)

            if saveAs_onnx :
                ModelFormat.save_as_onnx_model(outputModelPath)

        if stopAtSteps is not None and steps == stopAtSteps:
            exit()

def saveModel(model, outputModelPath):
    torch.save(model.state_dict(), outputModelPath)

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.eval()
    return model

def swap_face(model, target_face, source_face_latent):
    device = get_device()

    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)

    swapped_face = Image.postprocess_face(swapped_tensor)
    
    return swapped_face, swapped_tensor

# test image
test_img = ins_get_image('t1')

test_faces = faceAnalysis.get(test_img)
test_faces = sorted(test_faces, key = lambda x : x.bbox[0])
test_target_face, _ = face_align.norm_crop2(test_img, test_faces[0].kps, img_size)
test_target_face = Image.getBlob(test_target_face)
test_l = Image.getLatent(test_faces[2])

test_input = {inswapperInferenceSession.get_inputs()[0].name: test_target_face,
        inswapperInferenceSession.get_inputs()[1].name: test_l}

test_inswapperOutput = inswapperInferenceSession.run([inswapperInferenceSession.get_outputs()[0].name], test_input)[0]

def validate(modelPath):
    model = load_model(modelPath)
    swapped_face, swapped_tensor= swap_face(model, test_target_face, test_l)

    validation_content_loss, validation_identity_loss = style_loss_fn(swapped_tensor, torch.from_numpy(test_inswapperOutput).to(get_device()))

    validation_total_loss = validation_content_loss
    if validation_identity_loss is not None:
        validation_total_loss += validation_identity_loss

    return validation_total_loss, validation_content_loss, validation_identity_loss, swapped_face

def main():
    outputModelFolder = "model"
    modelPath = None
    # modelPath = f"{outputModelFolder}/reswapper-<step>.pth"

    logDir = "training/log"
    previewDir = "training/preview"
    datasetDir = "FFHQ"

    os.makedirs(outputModelFolder, exist_ok=True)
    os.makedirs(previewDir, exist_ok=True)

    train(
        datasetDir=datasetDir,
        model_path=modelPath,
        learning_rate=0.0001,

        outputModelFolder=outputModelFolder,
        saveModelEachSteps = 1000,
        stopAtSteps = 70000,
        logDir=f"{logDir}/{datetime.now().strftime('%Y%m%d %H%M%S')}",
        previewDir=previewDir)
                    
if __name__ == "__main__":
    main()