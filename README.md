# Attack on Object Detection via Contour
#### Yichi Zhang, Tsinghua University, zhangyic18@mails.tsinghua.edu.cn
## Introduction

This work is mainly for a competition held by Alibaba and CIKM2020 Workshop. The method is training the texture of the object contours to attack the detectors (Yolov4 and Faster RCNN). The final score is 3006 pts and the rank is 8th.


## Requirements

This code is based on pytorch. Some basic dependencies are recorded in `requirements.txt`

- torch
- torchvision
- pillow
- numpy
- tqdm
- scipy
- scikit-image
 
You can run yolov4 now if all above requirements are satisfied.

Another faster rcnn model is implemented based on mmdetection. So, ensure that the mmdetection library has been installed and can be run on your machine. You can refer install guide of mmdetection to [github](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md)

After installation, put the mmdetection directory into `./` below. Alternatively, it is optional that using [docker](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) provided by mmdetection.

## Usage

The folder `select1000_new` has the following structure:

```
|--select1000_new
    |-- XXX.png
    |-- XXX.png
    |-- XXX.png
    …
    |-- XXX.png
```

Before using, please notice that we have changed some source code of $mmdetection$. We modified 3 files under the directory `mmdetection/mmdet/` and all modification is labeld with `zyc`.

`GetOutline.py` is for acquiring the contour which is also the attack region of every image. `MyTrain.py` is for training the texture of the contour.
