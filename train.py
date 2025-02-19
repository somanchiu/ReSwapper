#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train.py
# Created Date: Monday December 27th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 22nd April 2022 10:49:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import time
import random
import argparse
import cv2
import numpy as np

import onnxruntime

import iresnet
onnxruntime.set_default_logger_severity(3)
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard

import Image
from util import util
# from util.plot import plot_batch

from models.projected_model import fsModel

# To do: clean code
from insightface.app import FaceAnalysis
import face_align

faceAnalysis = FaceAnalysis(name='buffalo_l', root='')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

# faceAnalysis_384 = FaceAnalysis(name='buffalo_l', root='')
# faceAnalysis_384.prepare(ctx_id=0, det_size=(384, 384))

faceAnalysis_256 = FaceAnalysis(name='buffalo_l', root='')
faceAnalysis_256.prepare(ctx_id=0, det_size=(256, 256))

faceAnalysis_128 = FaceAnalysis(name='buffalo_l', root='')
faceAnalysis_128.prepare(ctx_id=0, det_size=(128, 128))

faceAnalysisKV = {
    # '512': faceAnalysis,
    # '384': faceAnalysis_384,
    '256': faceAnalysis_256,
    '128': faceAnalysis_128,
}

def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')       

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

        # for training
        self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--load_optimizer', type=str2bool, default='True', help='load the optimizer')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr_g', type=float, default=0.0004, help='initial learning rate for adam - G')
        self.parser.add_argument('--lr_d', type=float, default=0.0004, help='initial learning rate for adam - D')

        self.parser.add_argument('--resize_image_to', type=int, default=512, help='resize the dataset images to a specific resolution')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss') 

        self.parser.add_argument("--arcface_model_path", type=str, default='arcface_w600k_r50.pth', help="arcface model path")
        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--sample_freq", type=int, default=1000, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=1000, help='frequence for saving the model')

        self.isTrain = True
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    log_path = os.path.join(opt.checkpoints_dir, opt.name, f'summary/{opt.lambda_id} {opt.lambda_feat} {opt.lambda_rec} {opt.lr_g} {opt.lr_d}')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))

    cudnn.benchmark = True

    model = fsModel()

    model.initialize(opt)

    #####################################################
    if opt.use_tensorboard:
        tensorboard_writer  = tensorboard.SummaryWriter(log_path)
        logger              = tensorboard_writer
        
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
        print('Resuming from epoch %d' % (start))        

    total_step  = opt.total_step

    #prepare validation image
    validation_img = cv2.imread('validationImage/t1.jpg')
    validation_faces = faceAnalysis.get(validation_img)
    validation_faces = sorted(validation_faces, key = lambda x : x.bbox[0])
    validation_target_face_256, _ = face_align.norm_crop2(validation_img, validation_faces[0].kps, 256)
    validation_target_face_256 = Image.getBlob(validation_target_face_256, (256, 256))
    validation_target_face_128, _ = face_align.norm_crop2(validation_img, validation_faces[0].kps, 128)
    validation_target_face_128 = Image.getBlob(validation_target_face_128, (128, 128))
    validation_source_latent = Image.getLatent(validation_faces[2])
    validation_target_latent = Image.getLatent(validation_faces[0])

    validation_target_face_256 = torch.from_numpy(validation_target_face_256).to(get_device())
    validation_target_face_128 = torch.from_numpy(validation_target_face_128).to(get_device())

    validation_target_latent = torch.from_numpy(validation_target_latent).to(get_device())
    validation_source_latent = torch.from_numpy(validation_source_latent).to(get_device())
    #

    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    from util.logo_class import logo_class
    logo_class.print_start_training()
    model.netD.feature_network.requires_grad_(False)

    opt.resize_image_to = 128

    from data.data_loader_Swapping import GetLoader

    train_loader    = GetLoader(opt.dataset,opt.batchSize,8,1234)

    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)

    mean = [0.485, 0.456, 0.406]  # RGB mean
    std = [0.229, 0.224, 0.225]   # RGB std
    from torchvision import transforms

    # Define preprocessing pipeline for tf_efficientnet_lite0
    preprocess = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    arcface = iresnet.iresnet50()
    arcface.load_state_dict(torch.load(opt.arcface_model_path, map_location=get_device()))
    arcface.to(get_device())
    arcface.eval()

    # Training Cycle
    for step in range(start, total_step):
        model.netG.train()
        
        interval = 0
        while interval <= 1 :
            try:
                loss_G_Rec = None

                random.shuffle(randindex)
                src_image1_d, src_image2_d  = train_loader.next()
                
                src_image1 = []
                src_image2 = []

                source_img = []

                src = [src_image1_d, src_image2_d]
                for srcIndex in range(len(src)):
                    images = src[srcIndex]
                    for index in range(len(images)):
                        tensor = images[index]
                        tensor = tensor.cpu()
                        tensor = tensor * 255

                        image = tensor.clamp(0, 255).numpy().astype(np.uint8)

                        # Convert from (C, H, W) to (H, W, C)
                        image = np.transpose(image, (1, 2, 0))
                        
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        targetFaceInfo = faceAnalysis.get(image)

                        target_face = targetFaceInfo[0]
                        aligned_target_face, M = face_align.norm_crop2(image, target_face.kps, opt.resize_image_to)
                        target_face_blob = Image.getBlob(aligned_target_face, (opt.resize_image_to, opt.resize_image_to))

                        if srcIndex == 0:
                            src_image1.append(target_face_blob[0])
                        else:
                            src_image2.append(target_face_blob[0])
            except:
                print("Next batch")
                continue

            src_image1 = torch.from_numpy(np.array(src_image1)).to(get_device())
            src_image2 = torch.from_numpy(np.array(src_image2)).to(get_device())

            latent_id = []
            targetEqSource = step%2 == 0
            # targetEqSource = random.random() < 0.5  # 50% chance for either case

            if targetEqSource:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]

            for img in img_id:
                img_id_112      = F.interpolate(img.unsqueeze(0),size=(112,112), mode='bicubic')
                latent_id.append(Image.getLatent_v2(arcface, (img_id_112 + 1)/2)[0])
            
            latent_id = torch.stack(latent_id, axis=0)
            # latent_id = torch.from_numpy(latent_id).to(get_device())

            if interval:
                # for i in range(1):
                img_fake        = model.netG(src_image1, latent_id)
                gen_logits,_    = model.netD(preprocess(img_fake.detach()), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = model.netD(preprocess(src_image2),None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(model.netD.parameters(), max_norm=1.0)
                optimizer_D.step()
            else:
                # for i in range(10):
                # model.netD.requires_grad_(True)
                img_fake        = model.netG(src_image1, latent_id)
                # G loss
                gen_logits,feat = model.netD(preprocess(img_fake), None)
                
                loss_Gmain      = (-gen_logits).mean()

                real_feat       = model.netD.get_feature(preprocess(src_image1))
                feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"]) 
                loss_G          = loss_Gmain + feat_match_loss * opt.lambda_feat
                
                try:
                    loss_G_ID = None

                    latent_fake = []
                    for img in img_fake:
                        # img_fake_img = Image.postprocess_face(img)
                        # cv2.imwrite("0.jpg", img_fake_img)
                        # fackFaceInfo = faceAnalysisKV[f'{opt.resize_image_to}'].get(img_fake_img)
                        output_112 = F.interpolate(img.unsqueeze(0), size=(112, 112), mode='bilinear', align_corners=False)

                        latent_fake.append(Image.getLatent_v2(arcface, (output_112 + 1 ) /2)[0])

                    # aligned_source_face, _ = face_align.norm_crop2(info['image'], info['info'].kps, image_size=112)

                    latent_fake = torch.stack(latent_fake, axis=0)


                    loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
                    loss_G += loss_G_ID * opt.lambda_id
                except Exception as e:
                    print(e)
                    print("Face not found")

                if targetEqSource:
                    #G_Rec, set this term to 0 if the source and target faces are from different identities
                    loss_G_Rec  = model.criterionRec(img_fake, src_image1) * opt.lambda_rec
                    loss_G      += loss_G_Rec

                optimizer_G.zero_grad()
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(model.netG.parameters(), max_norm=1.0)
                optimizer_G.step()
            
            interval += 1
        
        ############## Display results and errors ##########
        ### print out errors
        errors = {
            "G_Loss":loss_Gmain.item(),
            "G_feat_match":feat_match_loss.item(),
            "D_fake":loss_Dgen.item(),
            "D_real":loss_Dreal.item(),
            "D_loss":loss_D.item()
        }

        if loss_G_Rec is not None:
            errors["G_Rec"] = loss_G_Rec.item()

        if loss_G_ID is not None:
            errors["G_ID"] = loss_G_ID.item()
        if opt.use_tensorboard:
            for tag, value in errors.items():
                logger.add_scalar(tag, value, step)
        lossesMessage = 'Step: %d: ' % (step + 1)
        for k, v in errors.items():
            lossesMessage += '%s: %.3f ' % (k, v)

        print(lossesMessage)

        ### display output images
        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                output_128    = model.netG(validation_target_face_128, validation_source_latent)

                output_image = Image.postprocess_face(output_128)
                cv2.imwrite(os.path.join(sample_path, str(step+1)+'_128.jpg'), output_image)
                
                output_256    = model.netG(validation_target_face_256, validation_source_latent)

                output_image = Image.postprocess_face(output_256)
                cv2.imwrite(os.path.join(sample_path, str(step+1)+'_256.jpg'), output_image)
                # To do
                # plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

                print("Save preview")

        ### save latest model
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)            
            np.savetxt(iter_path, (step+1, total_step), delimiter=',', fmt='%d')

        # opt.resize_image_to += 128
        # if opt.resize_image_to > 256:
        #     opt.resize_image_to = 128
