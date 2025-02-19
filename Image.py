
import cv2
import numpy as np
import torch

emap = np.load("emap.npy")
input_std = 255.0
input_mean = 0.0

def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    return face_np

def getBlob(aimg, input_size = (128, 128)):
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size,
                            (input_mean, input_mean, input_mean), swapRB=True)
    return blob

def getLatent(source_face):
    latent = source_face.normed_embedding.reshape((1,-1))
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)

    return latent

def getLatent_v2(arcface, aligned_source_face):
    input_mean = 127.5
    input_std = 127.5
    input_size = (112, 112)
    emap_tensor = torch.from_numpy(emap).to('cuda')

    # aligned_source_face, _ = face_align.norm_crop2(source_img, faces2[0].kps, image_size=input_size[0])
    device = 'cuda'
    is_ndarray = isinstance(aligned_source_face, np.ndarray)
    if is_ndarray:
        blob = cv2.dnn.blobFromImages([aligned_source_face], 1.0 / input_std, input_size,
                                    (input_mean, input_mean, input_mean), swapRB=True)
        
        aligned_source_face = torch.from_numpy(blob).to(device)
    net_out = arcface(aligned_source_face)[0] # input shape [1,3,112,112] norm[-1,1]

    embedding = net_out.flatten()
    # from numpy.linalg import norm as l2norm
    # normed_embedding = embedding / l2norm(embedding)
    normed_embedding = embedding / torch.norm(embedding, p=2)
    # latent = 


    latent = torch.mm(normed_embedding.reshape((1, -1)), emap_tensor)
    latent = latent / torch.norm(latent)
    # latent = np.dot(latent, emap)
    # latent /= np.linalg.norm(latent)

    return latent

def blend_swapped_image(swapped_face, target_image, M):
    # get image size
    h, w = target_image.shape[:2]
    
    # create inverse affine transform
    M_inv = cv2.invertAffineTransform(M)
    
    # warp swapped face back to target space
    warped_face = cv2.warpAffine(
        swapped_face,
        M_inv,
        (w, h),
        borderValue=0.0
    )
    
    # create initial white mask
    img_white = np.full(
        (swapped_face.shape[0], swapped_face.shape[1]),
        255,
        dtype=np.float32
    )
    
    # warp white mask to target space
    img_mask = cv2.warpAffine(
        img_white,
        M_inv,
        (w, h),
        borderValue=0.0
    )
    
    # threshold and refine mask
    img_mask[img_mask > 20] = 255
    
    # calculate mask size for kernel scaling
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:  # safety check
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # erode mask
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        # blur mask
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # normalize mask
    img_mask = img_mask / 255.0
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    
    # blend images using mask
    result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
    result = result.astype(np.uint8)
    
    return result

def drawKeypoints(image, keypoints, colorBGR, keypointsRadius=2):
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=keypointsRadius, color=colorBGR, thickness=-1) # BGR format, -1 means filled circle