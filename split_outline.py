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


files = os.listdir("really_easy")

for file in files:
    if file=="4361.png":
        continue
    outline = cv2.imread("my_outline/"+file,cv2.IMREAD_GRAYSCALE)
    l=-1
    r=-1
    u=-1
    d=-1
    rowsum = np.sum(outline,axis = 1)
    colsum = np.sum(outline,axis = 0)
    row = np.where(rowsum>0)[0]
    col = np.where(colsum>0)[0]
    row_center = (int)((row.max()+row.min())/2)
    col_center = (int)((col.max()+col.min())/2)
    row_len = row.max()-row.min()
    col_len = col.max()-col.min()
    if row_len > col_len:
        offset = (int) (row_len*0.2)
        lm = row_center-offset
        rm = row_center+offset
        outline[lm:rm,:] = 0
    else:
        offset = (int) (col_len*0.2)
        um = col_center-offset
        dm = col_center+offset
        outline[:,um:dm] = 0
    # print(outline.shape)
    cv2.imwrite("really_outline/"+file,outline)

