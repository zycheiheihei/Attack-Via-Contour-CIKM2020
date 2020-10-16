import PIL
import cv2
import load_data
import copy
from tqdm import tqdm
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector,show_result_pyplot,get_Image_ready
import mmcv
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
from utils.utils import *
import numpy as np

import patch_config as patch_config
import sys
import time

# from brambox.io.parser.annotation import DarknetParser as anno_darknet_parse
from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
from skimage import measure

torch.cuda.set_device(3)

config = './mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
checkpoint = './models/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

MaskRCNN = init_detector(config, checkpoint, device='cpu').cuda()


img_dir = "select1000_new"
files = os.listdir(img_dir)


def masknms(output,nms_threshold = 0.5):
    bboxes = output[0]
    masks = output[1]
    for c in range(80):
        if bboxes[c].shape[0]==0:
            continue
        
        bboxes_class = bboxes[c]
        masks_class = masks[c]
        confidence = bboxes_class[:,4]
        # print(confidence)
        reverse_confidence = 1-confidence
        sort_index = np.argsort(reverse_confidence)
        # print(sort_index)
        for i in range(sort_index.shape[0]):
            id = sort_index[i]
            if bboxes_class[id][4]==0 or bboxes_class[id][4]<0.3:
                bboxes_class[id][4]=0
                continue
            for j in range(i+1,sort_index.shape[0]):
                compare_id = sort_index[j]
                boss_area = masks_class[id].sum()
                little_area = masks_class[id]*masks_class[compare_id]
                if little_area.sum()/boss_area > nms_threshold:
                    bboxes_class[compare_id][4]=0
        # print(bboxes_class)




count=0
dictation={}

for img in tqdm(files[:]):

    # print(img)
    img_path = os.path.join(img_dir,img)
    output = inference_detector(MaskRCNN,img_path)
    masknms(output)

    outline = torch.zeros((500,500))
    white = torch.ones((500,500))

    bboxes = np.concatenate(output[0])
    masks = list()
    for i in range(len(output[1])):
        if len(output[1][i])>0:
            for t in range(len(output[1][i])):
                output[1][i][t]=output[1][i][t][np.newaxis,:,:]
            output[1][i]=np.concatenate(output[1][i])
            masks.append(output[1][i])
    masks = np.concatenate(masks)

    old_outline = None


    for c in range(bboxes.shape[0]):
        if bboxes[c][4]>0.5:
            mask1 = masks[c]
            mask1 = np.array(mask1,dtype ='uint8')
            cv2_kernel = np.ones((3,3), np.uint8)
            masksmaller = cv2.erode(mask1,cv2_kernel,iterations=1)
            outline_temp = torch.tensor(mask1-masksmaller)
            old_outline = outline
            outline = torch.where(outline_temp>0,white,outline)
            if outline.sum()>4800:
                outline = old_outline
                break

    labels = measure.label(outline.numpy(), background=0, connectivity=1)
    labels = torch.tensor(labels)
    for i in range(labels.max()+1):
        black = torch.zeros(500,500)
        temp = torch.where(labels==i,white,black).sum()
        if temp < 250:
            black = torch.zeros(500,500)
            outline = torch.where(labels==i,black,outline)


    outline = outline.numpy()
    # print(outline.sum())
    if outline.sum()>5000:
        count+=1
    cv2.imwrite("outline_attack/result/"+img,outline*255)


print(count)



