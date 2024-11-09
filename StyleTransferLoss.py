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
        
        # Style loss
        self.style_loss = nn.MSELoss()
        
        # Face identity loss
        self.identity_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def gram_matrix(self, input):
        # a, b, c, d = input.size()
        # features = input.view(a * b, c * d)
        G = torch.mm(input, input.t())
        return G

    # def extract_face_embedding(self, image):
    #     # Convert torch tensor to numpy array
    #     face_tensor = image.squeeze().cpu().detach()
    #     # face_tensor = (face_tensor * 0.5 + 0.5).clamp(0, 1)
    #     face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #     face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    #     # Extract face embedding
    #     faces = self.face_analysis.get(face_np)
    #     if len(faces) == 0:
    #         return None
    #     return torch.tensor(faces[0].normed_embedding).to(self.device)
    
    def extract_face_latent(self, image):
        # Convert torch tensor to numpy array
        face_tensor = image.squeeze().cpu().detach()
        # face_tensor = (face_tensor * 0.5 + 0.5).clamp(0, 1)
        face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

        # Extract face embedding
        faces = self.face_analysis.get(face_np)
        if len(faces) == 0:
            return None
        return torch.tensor(Image.getLatent(faces[0])[0]).to(self.device)
    
    def get_style_loss(self, latent1, latent2):
        # target = torch.tensor([1.0]).to("cuda")
        similarity = torch.dot(latent1, latent2)
        
        # # Binary Cross-Entropy Loss
        # epsilon = 1e-7  # Small value to avoid log(0)
        # loss = -target * torch.log(similarity + epsilon) - (1 - target) * torch.log(1 - similarity + epsilon)
        
        return 1 - similarity

    def forward(self, output_image, target_content, target_face_latent, source_face_latent):
        # Content loss
        # content_loss = self.content_loss(output_image, target_content)
        content_loss = 1 - ssim(output_image, target_content, data_range=1.0)
 
        # Style loss
        # style_loss = 0
        # for out_feature, style_feature in zip(output_features, style_features):
        #     out_gram = self.gram_matrix(out_feature)
        #     style_gram = self.gram_matrix(style_feature)
        #     style_loss += self.style_loss(out_gram, style_gram)
        
        # Face identity loss

        output_embedding = self.extract_face_latent(output_image)
        target_embedding = self.extract_face_latent(target_content)

        identity_loss = None
        euclidean_distance = None
        
        if output_embedding is not None and target_embedding is not None:
            similarity = self.cosine_similarity(output_embedding, target_embedding)
            # similarity2 = self.cosine_similarity(output_embedding, torch.tensor(target_face_latent).to(self.device))
            # similarity2 = (similarity2 + 1) / 2
            identity_loss = 1-((similarity + 1) / 2)
            identity_loss = identity_loss ** 2 * 10
            # euclidean_distance = torch.sqrt(torch.sum((output_embedding - target_embedding) ** 2))
            # similarityA = self.cosine_similarity(output_embedding, output_embedding)
            # similarityB = self.cosine_similarity(target_embedding, target_embedding)

            # identity_loss +=similarity2
            # margin = 0.2
            # identity_loss = nn.functional.relu(margin - similarity)

            # target = torch.tensor([1.0]).to("cuda")
            # # Binary Cross-Entropy Loss
            # loss = -target * torch.log(similarity) - (1 - target) * torch.log(1 - similarity)
            
            # identity_loss= loss.mean()
            # identity_loss = 1 - self.identity_loss(output_embedding.unsqueeze(0), target_embedding.unsqueeze(0)).mean()
            # identity_loss = self.get_style_loss(output_embedding, target_embedding)
            # identity_loss = self.content_loss(output_embedding, target_embedding)
            # identity_loss = 1 - torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=0)
            # identity_loss = torch.tensor(0.0).to(self.device)

        # Total loss (you can adjust the weights as needed)
        # total_loss = content_loss*0.1 + identity_loss

        return content_loss, identity_loss, euclidean_distance

# Usage example:
# loss_fn = StyleTransferLoss()
# total_loss, content_loss, style_loss, identity_loss = loss_fn(content_features, style_features, output_image, target_content, target_style)