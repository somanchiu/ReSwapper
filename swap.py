import argparse
import os

import cv2
import torch
import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align

faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

from StyleTransferModel_128 import StyleTransferModel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments')
    
    parser.add_argument('--target', required=True, help='Target path')
    parser.add_argument('--source', required=True, help='Source path')
    parser.add_argument('--outputPath', required=True, help='Output path')
    parser.add_argument('--modelPath', required=True, help='Model path')

    return parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

def swap_face(model, target_face, source_face_latent, mask):
    device = get_device()

    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)
    
    swapped_face = Image.postprocess_face(swapped_tensor)
    
    return swapped_face, swapped_tensor

def main():
    args = parse_arguments()
    
    # Access the arguments
    target = args.target
    source = args.source
    output_path = args.outputPath
    model_path = args.modelPath

    model = load_model(model_path)

    target_image = cv2.imread(target)
    source_image = cv2.imread(source)

    target_face = faceAnalysis.get(target_image)[0]
    # faceAnalysis.prepare(ctx_id=0, det_size=(128, 128))

    source_face = faceAnalysis.get(source_image)[0]

    test_target_face, _ = face_align.norm_crop2(target_image, target_face.kps, 128)

    target_face_blob = Image.getBlob(test_target_face)
    source_latent = Image.getLatent(source_face)
    swapped_face, _ = swap_face(model, target_face_blob, source_latent, None)

    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(output_path, swapped_face)

if __name__ == "__main__":
    main()