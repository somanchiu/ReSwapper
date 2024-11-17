import cv2
import torch
import torch.nn as nn
import numpy as np
from insightface.app import FaceAnalysis
from pytorch_msssim import ssim

import Image

class StyleTransferLoss(nn.Module):
    def __init__(self, device='cuda', face_analysis = None):
        super(StyleTransferLoss, self).__init__()
        if face_analysis is None:
            self.face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_analysis.prepare(ctx_id=0, det_size=(128, 128))
        else:
            self.face_analysis = face_analysis
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=0)
        
        # Content loss
        self.content_loss = nn.MSELoss()
    
    def extract_face_latent(self, image):
        # Convert torch tensor to numpy array
        face_tensor = image.squeeze().cpu().detach()
        face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

        # Extract face embedding
        faces = self.face_analysis.get(face_np)
        if len(faces) == 0:
            return None
        return torch.tensor(Image.getLatent(faces[0])[0]).to(self.device)
    
    def forward(self, output_image, target_content):
        # Content loss
        # content_loss = self.content_loss(output_image, target_content)
        content_loss = 1 - ssim(output_image, target_content, data_range=1.0)
 
        output_embedding = self.extract_face_latent(output_image)
        target_embedding = self.extract_face_latent(target_content)

        identity_loss = None
        euclidean_distance = None
        
        if output_embedding is not None and target_embedding is not None:
            similarity = self.cosine_similarity(output_embedding, target_embedding)

            identity_loss = 1-((similarity + 1) / 2)
            identity_loss = identity_loss ** 2 * 10

        return content_loss, identity_loss, euclidean_distance
