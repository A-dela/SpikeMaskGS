#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim, msssim
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import imageio
import os
import numpy as np
import cv2
import csv
import torch 
import lpips as lp
from utils.image_utils import psnr
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from skimage.metrics import structural_similarity as msssim  # 引入 msssim


tonemap = lambda x : (np.log(np.clip(x, 0, 1) * 5000 + 1 ) / np.log(5000 + 1)).astype(np.float32)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



def readImages(renders_dir, gt_dir, dyn_gt_dir=None):
    renders = []
    gts = []
    image_names = []
    # if dyn_gt_dir is not None:
    #     dyn_gts = []
    for fname in os.listdir(renders_dir):
        render = Image.open( os.path.join(renders_dir , fname))
        gt = Image.open(os.path.join(gt_dir , fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        # if dyn_gt_dir is not None:
        #     dyn_gt = Image.open(os.path.join(dyn_gt_dir, fname))
        #     dyn_gts.append(tf.to_tensor(dyn_gt).unsqueeze(0)[:, :1, :, :].cuda())
        image_names.append(fname)
    # if dyn_gt_dir is not None:
    #     return renders, gts,  None, image_names
    # else:
    return renders, gts, image_names

def evaluate_(test_path, iteration):

    test_path_list = [os.path.join(test_path, "test", f) for f in sorted(os.listdir(os.path.join(test_path, "test"))) if 'ours' in f and str(iteration) in f]
    print(f"Found test models:", test_path_list)
    # test_path = test_path_list[-1]
    test_path = test_path_list[0]
    print(f"Selected test model is: {test_path}")

    dssim2s = []
    dssim1s = []
    psnrs = []
    lpipss = []
    dy_psnrs = []
    dy_lpipss = []
    gt_dir = os.path.join(test_path, "gt")
    renders_dir =os.path.join(test_path , "renders")

    renders, gts, image_names = readImages(renders_dir, gt_dir)
    csvfile = open(os.path.join(test_path, 'eval.csv'),"w") 
    writer = csv.writer(csvfile)
    
    if "vru" in renders_dir:
        network = 'vgg'
    else:
        network = 'alex'

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

        dssim1 =  (1 - ssim(renders[idx], gts[idx])) / 2
        # dssim2  =  (1-  msssim(renders[idx], gts[idx])) / 2
        
        psnr_ = psnr(renders[idx], gts[idx])
        # todo check dataset type for lpips model
        lpips_ = lpips(renders[idx], gts[idx], net_type=network) # vgg for VRU, alex for other datasets
        # lpips_ = lpips(renders[idx], gts[idx], net_type='alex') # dynerf

        dssim1s.append(dssim1)
        # dssim2s.append(dssim2)
        psnrs.append(psnr_)
        lpipss.append(lpips_)  # this used vgg model
        # writer.writerow([ idx , psnr_.item(), dssim1.item(), dssim2.item(), lpips_.item()])
        writer.writerow([ idx , psnr_.item(), dssim1.item(), lpips_.item()])
    
    avg_ssim = torch.tensor(dssim1s).mean().item()
    # avg_msssim = torch.tensor(dssim2s).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr, ".5"))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_ssim, ".5"))
    # print(" Avg DSSIM2 : {:>12.7f}".format(avg_msssim, ".5"))
    print(" Avg LPIPS: {:>12.7f}".format(avg_lpips, ".5"))
    print("")

    writer.writerow([ "avg" , avg_psnr, avg_ssim, avg_lpips])



if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Mlp = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    evaluate_(args.model_path, args.iteration)

