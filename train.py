from datetime import datetime
import os
import random
import onnx
import torch
import torch.nn as nn
import torch.optim as optim

import Image
from StyleTransferLoss import StyleTransferLoss
import onnxruntime as rt

import numpy as np
import cv2
from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from StyleTransferModel_128 import StyleTransferModel

# from facexlib.utils.face_restoration_helper import FaceRestoreHelper

# # from PerceptualLoss import PerceptualLoss

# face_helper = FaceRestoreHelper(
#             upscale_factor=1,
#             face_size=128,
#             crop_ratio=(1, 1),
#             det_model='retinaface_resnet50',
#             save_ext='png',
#             device="cuda",
#         )
inswapper_128_path = 'inswapper_128.onnx'
img_size = 128

logDir = None
# logWriter = None

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = rt.InferenceSession(inswapper_128_path, providers=providers)
# from pulid.pipeline import PuLIDPipeline
# import pulid.utils

import torch.nn.functional as F

# from VGGFeatureExtractor import VGGFeatureExtractor

# puLID_pipeline = PuLIDPipeline()

# from MaskedImage import MaskedImage
# maskedImageCreator = MaskedImage()

from torch.utils.tensorboard import SummaryWriter

faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(640, 640))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_loss_fn = StyleTransferLoss().to(get_device())

# facial_mask = Mask()

# 1. Define the transformer model
# import torch.nn.functional as F
# from torchvision import transforms, models

def create_onnx_model(torch_model_path, save_emap=True):
    output_path = torch_model_path.replace(".pth", ".onnx")

    device = get_device()
    # Initialize model with the pretrained weights
    torch_model = StyleTransferModel().to(device)
    torch_model.load_state_dict(torch.load(torch_model_path, map_location=device))

    # set the model to inference mode
    torch_model.eval()
    torch.onnx.export(torch_model,               # model being run
                  {
                      'target' :torch.randn(1, 3, img_size, img_size, requires_grad=True).to(device), 
                      'source': torch.randn(1, 512, requires_grad=True).to(device),
                    #   'mask': torch.randn(1, 128, 128, requires_grad=True).to(device)
                  },                         # model input (or a tuple for multiple inputs)
                  output_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['target', "source"],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    model = onnx.load(output_path)

    if save_emap :
        emap = np.load("emap.npy")

        emap_tensor = onnx.helper.make_tensor(
            name='emap',
            data_type=onnx.TensorProto.FLOAT,
            dims=[512, 512],
            vals=emap
        )
        
        model.graph.initializer.append(emap_tensor)
        
        onnx.save(model, output_path)

    # printable_graph=onnx.helper.printable_graph(model.graph)
    # f = open("test.onnx.helper.printable_graph.txt", "w")
    # f.write(printable_graph)
    # f.close()

def masked_loss(output, face, mask, criterion, passMaskToCriterion=False):
    masked_output = output * mask
    masked_face = face * mask
    if passMaskToCriterion == False:
        loss = criterion(masked_output, masked_face)
    else:
        loss = criterion(masked_output, masked_face, mask)
    return loss / (mask.sum() + 1e-8) # Normalize by the number of masked pixels

def background_regularization(output, target, mask):
    background_diff = (output - target) * mask
    return torch.mean(torch.abs(background_diff))

def pixel_wise_loss(output, swapped_face):
    """
    Computes the mean squared error between each pixel of the output and swapped_face.
    
    Args:
    output (torch.Tensor): The output image tensor from your model (B, C, H, W)
    swapped_face (torch.Tensor): The ground truth swapped face tensor (B, C, H, W)
    
    Returns:
    torch.Tensor: The mean squared error loss
    """
    return nn.functional.mse_loss(output, swapped_face)

def mse_loss(output, target):    
    # Normalize pixel values to [0, 1] if they aren't already
    if output.max() > 1.0 or target.max() > 1.0:
        output = output / 255.0
        target = target / 255.0
    
    # Reshape tensors to [batch_size, channels, height, width]
    output = output.permute(0, 1, 2)
    target = target.permute(0, 1, 2)
    
    # Calculate MSE loss
    loss = F.mse_loss(output, target)
    
    return loss

import numpy as np

def compute_cyclic_consistency_loss(model, target, source, output):
    # Forward pass
    # output = model(target, source)
    
    # Reverse pass
    reconstructed_target, reconstructed_latent = model.reverse_style_transfer(output)
    
    # Compute cyclic consistency losses
    image_consistency_loss = F.mse_loss(reconstructed_target, target)
    latent_consistency_loss = F.mse_loss(reconstructed_latent, source)
    
    return image_consistency_loss, latent_consistency_loss

def calculate_lstyle(IR, IT, G_phi):
    """
    Calculate the style loss (Lstyle) between a reference image and a test image.
    
    Parameters:
    IR (list of np.array): Gram matrices of reference image features at different layers
    IT (list of np.array): Gram matrices of test image features at different layers
    G_phi (callable): Function to compute Gram matrix (if not pre-computed)
    
    Returns:
    float: The calculated style loss
    """
    lstyle = 0
    
    for i in range(len(IR)):
        # If IR and IT are not pre-computed Gram matrices, compute them
        if G_phi is not None:
            gram_IR = G_phi(IR[i])
            gram_IT = G_phi(IT[i])
        else:
            gram_IR = IR[i]
            gram_IT = IT[i]
        
        # Calculate the Frobenius norm of the difference
        diff_norm = torch.norm(gram_IR - gram_IT, p='fro')
        
        # Add to the total style loss
        lstyle += diff_norm
    
    return lstyle

# Example usage:
# Assuming IR and IT are lists of feature maps or pre-computed Gram matrices

def gram_matrix(feature_map):
    """
    Compute the Gram matrix of a feature map.
    """
    height, width = feature_map.size()
    feature_map = feature_map.view(1, 1, height * width)
    return torch.bmm(feature_map, feature_map.transpose(1, 2))

def cosine_similarity_loss(output, target):
    return 1 - F.cosine_similarity(output, target, dim=1).mean()

def color_consistency_loss(output, target):
    output_mean = torch.mean(output, dim=[2, 3])
    target_mean = torch.mean(target, dim=[2, 3])
    return F.mse_loss(output_mean, target_mean) * 100

def mrf_loss(output, target, window_size=5):
    """
    Compute MRF loss between output and target images.
    """
    def extract_patches(x):
        return F.unfold(x, kernel_size=window_size, stride=1, padding=window_size//2)
    
    output_patches = extract_patches(output)
    target_patches = extract_patches(target)
    
    output_patches = output_patches.permute(0, 2, 1)
    target_patches = target_patches.permute(0, 2, 1)
    
    dist = torch.cdist(output_patches, target_patches)
    min_dist, _ = torch.min(dist, dim=2)
    
    return torch.mean(min_dist)

# linear = []
   
# def load_styleBlockWB():
#     styleWBdir = "style_wb"

#     for nodeIndex in range(0, 6):
#         style_block_1_2 = []
#         for index in range(1, 3):
#             new_linear=nn.Linear(512, 2048).to("cuda")
#             styleBlockWB = {
#                 'weight': (np.load(f'{styleWBdir}/styles_{nodeIndex}_style{index}_linear_weight.npy')),
#                 'bias': (np.load(f'{styleWBdir}/styles_{nodeIndex}_style{index}_linear_bias.npy'))
#             }
#             with torch.no_grad():
#                 new_linear.weight.copy_(torch.FloatTensor(styleBlockWB["weight"]))
#                 new_linear.bias.copy_(torch.FloatTensor(styleBlockWB["bias"]))
#             style_block_1_2.append(new_linear)
#         linear.append(style_block_1_2)

# load_styleBlockWB()

def f32tof16tof32(tensor_f32):
    tensor_f16 = tensor_f32.to(torch.float16)
    
    # Convert back to float32
    tensor_f32 = tensor_f16.to(torch.float32)

    return tensor_f32

# def process_conv_layers(modelStyle, source):
#     style_loss = None
#     softmax = nn.Softmax().to("cuda")
#     mse=nn.MSELoss().to("cuda")
#     for nodeIndex in range(0, 6):
#         for index in range(0, 2):
#             y = linear[nodeIndex][index](source)[0]
#             # y=f32tof16tof32(y)
#             y = torch.where(torch.isnan(y), torch.tensor(float('-inf'), device=y.device), y)
#             x = modelStyle[nodeIndex][index]
#             # x=f32tof16tof32(x)

#             x = torch.where(torch.isnan(x), torch.tensor(float('-inf'), device=x.device), x)[0]

#             newX=[]
#             newY=[]
#             for xyIndex in range(0, 2048):
#                 if torch.isnan(x[xyIndex]) | torch.isinf(x[xyIndex]) | torch.isnan(y[xyIndex]) | torch.isinf(y[xyIndex]):
#                     continue
                
#                 newX.append(x[xyIndex])
#                 newY.append(y[xyIndex])

#             temp_style_loss = mse(torch.tensor(newX).to("cuda"), torch.tensor(newY).to("cuda"))

#             # max_value = 1e-20  # You can adjust this value as needed
#             # temp_style_loss = torch.where(torch.isnan(temp_style_loss) | torch.isinf(temp_style_loss), torch.tensor(max_value, device=temp_style_loss.device), temp_style_loss)

#             if not torch.isnan(temp_style_loss) and not torch.isinf(temp_style_loss):
#                 if style_loss == None:
#                     style_loss = temp_style_loss
#                 else:
#                     style_loss += temp_style_loss

#     # style_loss = torch.where(torch.isnan(style_loss), torch.tensor(float(1), device=style_loss.device), style_loss)

#     return style_loss

def train(datasetDir, dataset=None, num_epochs=1000, batch_size=1, learning_rate=0.0001, model_path=None, outputModelFolder='', saveModelEachSteps = 1, stopAtSteps=None, logDir=None, previewDir=None, saveAs_onnx = False):
    # Get the device (GPU if available, else CPU)
    device = get_device()
    print(f"Using device: {device}")

    model = StyleTransferModel().to(device)# Move model to GPU

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

        lastSteps = int(model_path.split('-')[-1].split('.')[0])
        print(f"Resuming training from step {lastSteps}")
    else:
        lastSteps = 0
        
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.MSELoss()
    # perceptual_criterion = PerceptualLoss().to(device)
    # Initialize model, loss function, and optimizer
    model.train()
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, verbose=True)
    # perceptual_criterion = PerceptualLoss().to(device)
    # vgg_extractor = VGGFeatureExtractor().to(device)
    # criterion = StyleTransferLoss(content_weight=1.0, style_weight=1.0, tv_weight=1e-4).to(device)
    # style_loss_fn = StyleLoss().to(device)

    # Initialize TensorBoard writer
    if logDir is not None:
        train_writer = SummaryWriter(os.path.join(logDir, "training"))
        val_writer = SummaryWriter(os.path.join(logDir, "validation"))

    steps = 0

    # datasetDir = "training/dataset/v7"

    # aligned_face = list(np.load(f'{datasetDir}/aligned_face.npy'))
    # face_latent = list(np.load(f'{datasetDir}/face_latent.npy'))

    modelInputs=sess.get_inputs()
    modeloutputs = sess.get_outputs()
    input_name = sess.get_inputs()[0].name
    output_name = modeloutputs[0].name
    epoch=0

    # feature_extractor = FeatureExtractor().to(device).eval()

    image = os.listdir(datasetDir)

    alpha=0.01
    temperature=2.0

    # Training loop
    while True:
        start_time = datetime.now()
        # currentEpoch = epoch + 1
        # print(f"CurrentEpoch: {currentEpoch}")


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

        # blob = inswapper.getBlob(aligned_face[targetFaceIndex])
        # latent = face_latent[sourceFaceIndex]

        if targetFaceIndex != sourceFaceIndex:
            input = {sess.get_inputs()[0].name: blob,
                    sess.get_inputs()[1].name: latent}

            target_output = sess.run([output_name], input)[0]
        else:
            target_output = blob
        # for source_face, source_face_latent, target_face, swapped_face, mask in dataloader:
        # source_face = source_face.to(device)
        # source_face_latent = source_face_latent.to(device)
        # target_face = target_face.to(device)
        # swapped_face = swapped_face.to(device)
        # mask = mask.to(device)
        latent_tensor = torch.from_numpy(latent).to(device)
        target_input_tensor = torch.from_numpy(blob).to(device)

                
        # Compute  features
        # masked_target_face = maskedImageCreator.create(postprocess_face(target_face))
        optimizer.zero_grad()
        output = model(target_input_tensor, latent_tensor)

        # style_loss = process_conv_layers(style, latent_tensor)
        # training_model_style=style[0][0]
        # softmax = nn.Softmax().to(device)
        # mse=nn.MSELoss().to(device)
        # style_loss = mse(softmax(w), softmax(training_model_style))
        # output_face_img=postprocess_face(output)
        # target_face_img=postprocess_face(torch.from_numpy(target_output).to(device))

        # with torch.no_grad():
        #     output_features, _ = feature_extractor(feature_extractor.load_image(output_face_img).to(device))
        #     _, style_features = feature_extractor(feature_extractor.load_image(target_face_img).to(device))
        target_tensor = torch.from_numpy(target_output).to(device)
        # target_face_latent = inswapper.getLatent(faces[0])[0]

        content_loss, identity_loss, euclidean_distance = style_loss_fn(output, target_tensor, None, latent_tensor[0])
        # image_cycle_loss, latent_cycle_loss = compute_cyclic_consistency_loss(model, target_input_tensor, latent_tensor, output)
        # if loss is None:
        #     continue

        # soft_targets = F.softmax(target_tensor / temperature, dim=1)
        # soft_prob = F.log_softmax(output / temperature, dim=1)

        # # Calculate distillation loss
        # distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

        # gradient_loss_val = gradient_loss(output, target_tensor)

        # perceptual_loss = perceptual_criterion(output, target_tensor)
        # loss = content_loss + 0.1 * image_cycle_loss + 0.1 * latent_cycle_loss
        loss = content_loss
        # loss = content_loss + perceptual_loss

        if identity_loss is not None:
            loss +=identity_loss

        if euclidean_distance is not None:
            loss +=euclidean_distance

        # loss +=gradient_loss_val

        # if not torch.isnan(style_loss) and not torch.isinf(style_loss):
        #     loss +=style_loss

        # face_helper.read_image(output_face_img)
        # face_helper.read_image(target_face_img)
        # face_helper.get_face_landmarks_5(only_center_face=True)
        # face_helper.align_warp_face()

        # if len(face_helper.cropped_faces) >0:
        #     faces = faceAnalysis.get(face_helper.cropped_faces[0])
        # face_helper.clean_all()
        # face_helper.read_image(output_face_img)
        # face_helper.get_face_landmarks_5(only_center_face=True)
        # face_helper.align_warp_face()
        # if len(face_helper.cropped_faces) >0:
        #     faces2 = faceAnalysis.get(face_helper.cropped_faces[0])
        # face_helper.clean_all()
        # cv2.imwrite("test/cropped_faces.png", face_helper.cropped_faces[0])
        # cv2.imwrite("test/target_face_img.png", target_face_img)
        # x= cv2.imread("test/output_face_img.png")

        # faceAnalysis.prepare(ctx_id=0, det_size=(80, 80), det_thresh=0.3)

        # faces2 = faceAnalysis.get(target_face_img)

        # style_loss = None
        # landmark_loss = None
        # if len(faces)>0 and len(faces2)>0:
        #     latent1 =inswapper.getLatent(faces[0])[0]
        #     latent2 =inswapper.getLatent(faces2[0])[0]
        #     latent1=torch.from_numpy(latent1).to(device)
        #     latent2=torch.from_numpy(latent2).to(device)

        #     style_loss = F.mse_loss(latent1, latent2)
        #     # embedding1 = torch.from_numpy(faces[0].normed_embedding).to(device)
        #     # embedding2=torch.from_numpy(faces2[0].normed_embedding).to(device)
        #     # style_loss = 1 - torch.dot(embedding1, embedding2).mean()

        #     # landmark_2d_106 = torch.from_numpy(faces[0].landmark_2d_106).to(device)
        #     # landmark_2d_106_2 = torch.from_numpy(faces2[0].landmark_2d_106).to(device)
            
        #     # landmark_loss = F.mse_loss(landmark_2d_106, landmark_2d_106_2)
        # else:
        #     continue
            # inswapper.getLatent()

        # cv2.imwrite("test/output.png", test)

        # style_loss = style_loss_fn(output, swapped_face)

        # saveModel(model, outputModelFolder, 1)
        # cv2.imwrite("test/output.png", postprocess_face(output))
        # #1

        # cv2.imwrite("test/swapped_face.png", swapped_face_img)
        # loss, content_loss, style_loss, tv_loss = criterion(
        #     torch.from_numpy(output_face_img.astype(np.float32)).permute(2, 0, 1).to(device).unsqueeze(0),
        #     torch.from_numpy(postprocess_face(target_face).astype(np.float32)).permute(2, 0, 1).to(device).unsqueeze(0),
        #     torch.from_numpy(postprocess_face(swapped_face).astype(np.float32)).permute(2, 0, 1).to(device).unsqueeze(0),
        #     vgg_extractor)

        # id_mask = 1 - mask
        # output_face_img = postprocess_face(output)
        # swapped_face_img = postprocess_face(swapped_face)

        # # cv2.imwrite("test/swapped_face.png", swapped_face_img)
        # style_loss = None

        # output_face = faceAnalysis.get(output_face_img)
        # swapped_face_an = faceAnalysis.get(swapped_face_img)

        # if len(output_face) > 0 and len(swapped_face_an) > 0:
        #     face = output_face[0]
        #     face_normed_embedding= torch.from_numpy(face.normed_embedding).to(device).unsqueeze(0)
        #     output_latent=inswapper.getLatent(face)
        #     output_latent = torch.from_numpy(output_latent).to(device)

        #     face2 = swapped_face_an[0]
        #     swapped_face_normed_embedding= torch.from_numpy(face2.normed_embedding).to(device).unsqueeze(0)

        #     swapped_face_latent = inswapper.getLatent(face2)
        #     swapped_face_latent = torch.from_numpy(swapped_face_latent).to(device)
        #     cosine_similarity_style_loss = cosine_similarity_loss(face_normed_embedding, swapped_face_normed_embedding)
        #     style_loss = cosine_similarity_style_loss
        #     # style_loss = calculate_lstyle([face_normed_embedding], [swapped_face_normed_embedding], gram_matrix)
        #     # cosine_similarity_style_loss = F.mse_loss(output_latent, swapped_face_latent)

        #     # style_loss = criterion.style_loss(output_latent, source_face_latent.unsqueeze(2).unsqueeze(0))

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test.png", img)
        # output_face_img = postprocess_face(output)
        # swapped_face_img = postprocess_face(swapped_face)
        # cv2.imwrite("test/swapped_face.png", swapped_face_img)

        # resized_img =  pulid.utils.resize_numpy_image_long(output_face_img, 1024)
        # output_face_id_embedding = puLID_pipeline.get_id_embedding(resized_img, True)
        # if output_face_id_embedding is not None:
        #     output_face_id_embedding= output_face_id_embedding.unsqueeze(0)

        # # resized_img =  pulid.utils.resize_numpy_image_long(swapped_face_img, 1024)
        # temp = source_face.squeeze().cpu().detach().numpy()
        # # cv2.imwrite("test/swapped_face.png", temp)
        # resized_img =  pulid.utils.resize_numpy_image_long(temp, 1024)

        # swapped_face_id_embedding = puLID_pipeline.get_id_embedding(resized_img, True)
        # if swapped_face_id_embedding is not None:
        #     swapped_face_id_embedding = swapped_face_id_embedding.unsqueeze(0)


        # content_loss = F.mse_loss(output, swapped_face)*0.5 + F.mse_loss(output*id_mask, swapped_face*id_mask) + F.mse_loss(output*mask, target_face*mask) + ((1-F.mse_loss(output*id_mask, target_face*id_mask)))
        # content_loss = F.mse_loss(output, torch.from_numpy(target_output).to(device))
        # loss = content_loss
        # if style_loss is not None:
        #     loss = loss + style_loss

        # if landmark_loss is not None:
        #     loss = loss + landmark_loss
        # if cosine_similarity_style_loss is not None:
        #     loss = loss + cosine_similarity_style_loss

        # loss, content_loss, style_loss, tv_loss = criterion(output, target_face, source_face, swap_face, vgg_extractor, mask)
        
        # loss = loss * 0.5 + source_face_mse_loss * 0.5
        # Backpropagation
        # loss.requires_grad = True
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # scheduler.step(total_loss)

        steps += 1
        totalSteps = steps + lastSteps

        # current_lr = scheduler.get_last_lr()[0]

        if logDir is not None:
            train_writer.add_scalar("Loss/total", loss.item(), totalSteps)
            # Log the loss to TensorBoard
            # logWriter.add_scalar('Loss/total', loss.item(), totalSteps)
            # logWriter.add_scalar('Loss/distillation_loss', distillation_loss.item(), totalSteps)
            train_writer.add_scalar("Loss/content_loss", content_loss.item(), totalSteps)

            # logWriter.add_scalar('Loss/content_loss', content_loss.item(), totalSteps)
            # logWriter.add_scalar('Loss/image_cycle_loss', image_cycle_loss.item(), totalSteps)
            # logWriter.add_scalar('Loss/latent_cycle_loss', latent_cycle_loss.item(), totalSteps)

            # logWriter.add_scalar('Loss/gradient_loss_val', gradient_loss_val.item(), totalSteps)
            # if perceptual_loss is not None:
            #     logWriter.add_scalar('Loss/perceptual_loss', perceptual_loss.item(), totalSteps)

            if identity_loss is not None:
                train_writer.add_scalar("Loss/identity_loss", identity_loss.item(), totalSteps)

            # if euclidean_distance is not None:
            #     logWriter.add_scalar('Loss/euclidean_distance', euclidean_distance.item(), totalSteps)

            # if style_loss is not None:
            #     logWriter.add_scalar('Loss/style_loss', style_loss.item(), totalSteps)
            # if cosine_similarity_style_loss is not None:
            #     logWriter.add_scalar('Loss/cosine_similarity_style_loss', cosine_similarity_style_loss.item(), totalSteps)
            # logWriter.add_scalar('Loss/tv_loss', tv_loss.item(), totalSteps)
            # if len(output_face) > 0:
            #     logWriter.add_scalar('Loss/source_face_mse_loss', source_face_mse_loss.item(), totalSteps)
            # logWriter.add_scalar('Learning_rate', current_lr, totalSteps)

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

            val_writer.add_scalar("Loss/total", validation_total_loss.item(), totalSteps)
            val_writer.add_scalar("Loss/content_loss", validation_content_loss.item(), totalSteps)
            if validation_identity_loss is not None:
                val_writer.add_scalar("Loss/identity_loss", validation_identity_loss.item(), totalSteps)

            if saveAs_onnx :
                create_onnx_model(outputModelPath)

        if stopAtSteps is not None and steps == stopAtSteps:
            exit()

def saveModel(model, outputModelPath):
    torch.save(model.state_dict(), outputModelPath)

def distillation_loss(teacher_output, student_output, temperature=2.0):
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_output / temperature, dim=1),
        torch.nn.functional.softmax(teacher_output / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# def preprocess_face(face_img):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     return transform(face_img).unsqueeze(0)

def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    # face_tensor = (face_tensor * 0.5 + 0.5).clamp(0, 1)
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    # img_fake = face_tensor.transpose((0,2,3,1))[0]
    # bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]

    return face_np

def swap_face(model, target_face, source_face_latent, mask):
    device = get_device()

    # mask_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # Preprocess source face and target face

    # source_face_latent = inswapper.getLatent(source_face)
    # target_face = inswapper.getBlob(target_face)

    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)
    # mask_tensor = mask_transform(mask).unsqueeze(0).to(device)
    
    # Generate swapped face
    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)
    
    # Postprocess the result
    swapped_face = postprocess_face(swapped_tensor)
    
    return swapped_face, swapped_tensor

# test image
test_img = ins_get_image('t1')

test_faces = faceAnalysis.get(test_img)
test_faces = sorted(test_faces, key = lambda x : x.bbox[0])
# test_source_face, _ = face_align.norm_crop2(test_img, test_faces[2].kps, 128)
# test_source_face2 = faceAnalysis.get(test_source_face)[0]
test_target_face, _ = face_align.norm_crop2(test_img, test_faces[0].kps, img_size)
test_target_face = Image.getBlob(test_target_face)
test_l = Image.getLatent(test_faces[2])

test_input = {sess.get_inputs()[0].name: test_target_face,
        sess.get_inputs()[1].name: test_l}

test_inswapperOutput = sess.run([sess.get_outputs()[0].name], test_input)[0]
# test_target_face2 = faceAnalysis.get(test_target_face)[0]
# test_target_face2, _ = face_align.norm_crop2(test_target_face, test_target_face2.kps, 128)
# test_target_face, _ = face_align.norm_crop2(test_target_face, test_target_face2[0].kps, 128)

# Create face mask
# mask = facial_mask.create_face_mask(test_target_face) 
# mask = (mask == 0).astype(int)
# mask = 1 - mask

# if mask is not None:
    # Apply mask to image
# test_source_face = facial_mask.apply_mask(test_source_face, mask)
# test_target_face = facial_mask.apply_mask(test_target_face, mask, True)
# test_swaped_face, _ = insw(test_target_face, test_target_face2[0].kps, 128)

# cv2.imwrite("test/test_source_face.png", test_source_face)
# cv2.imwrite("test/test_target_face.png", test_target_face2)
# cv2.imwrite("test/test_masked_target_face.png", test_masked_target_face)

def gradient_loss(output, target):
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    output_dx, output_dy = gradient(output)
    target_dx, target_dy = gradient(target)
    
    grad_diff_x = torch.abs(output_dx - target_dx)
    grad_diff_y = torch.abs(output_dy - target_dy)
    
    return torch.mean(grad_diff_x) + torch.mean(grad_diff_y)

def validate(modelPath):
    model = load_model(modelPath)
    swapped_face, swapped_tensor= swap_face(model, test_target_face, test_l, None)

    validation_content_loss, validation_identity_loss, _ = style_loss_fn(swapped_tensor, torch.from_numpy(test_inswapperOutput).to(get_device()), None, None)

    validation_total_loss = validation_content_loss
    if validation_identity_loss is not None:
        validation_total_loss += validation_identity_loss

    return validation_total_loss, validation_content_loss, validation_identity_loss, swapped_face

# 5. Main function to run the training
def main():
    outputModelFolder = "model"
    modelPath = None
    # modelPath = f"{outputModelFolder}/reswapper-429500.pth"

    logDir = "training/log"
    previewDir = "training/preview"
    datasetDir = "FFHQ"

    os.makedirs(previewDir, exist_ok=True)

    train(
        datasetDir=datasetDir,
        model_path=modelPath,
        # dataset=dataset,
        # learning_rate=0.0000001,
        learning_rate=0.0001,

        outputModelFolder=outputModelFolder,
        saveModelEachSteps = 1000,
        stopAtSteps = 70000,
        logDir=f"{logDir}/{datetime.now().strftime('%Y%m%d %H%M%S')}",
        previewDir=previewDir)
                    
if __name__ == "__main__":
    main()