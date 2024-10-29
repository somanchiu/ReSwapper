
import cv2
import numpy as np

emap = np.load("emap.npy")
input_std = 255.0
input_mean = 0.0
input_size = (128, 128)

def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    return face_np

def getBlob(aimg):
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size,
                            (input_mean, input_mean, input_mean), swapRB=True)
    return blob

def getLatent(source_face):
    latent = source_face.normed_embedding.reshape((1,-1))
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)

    return latent