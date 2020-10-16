"""
Training code for Adversarial patch training using Faster RCNN based on mmdetection


"""

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
csv_name = 'x_result2.csv'
torch.cuda.set_device(5)

def patch_aug(input_tensor):

    ## patch augementation
    min_contrast = 0.8
    max_contrast = 1.2
    min_brightness = -0.1
    max_brightness = 0.1
    noise_factor = 0.10
    batch_size = input_tensor.shape[0]
    adv_batch = input_tensor.unsqueeze(0)

    # Create random contrast tensor
    contrast = torch.Tensor(batch_size).uniform_(min_contrast, max_contrast)
    contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contrast = contrast.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    # Create random brightness tensor
    brightness = torch.Tensor(batch_size).uniform_(min_brightness, max_brightness)
    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    brightness = brightness.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    # Create random noise tensor
    noise = torch.Tensor(adv_batch.size()).uniform_(-1, 1) * noise_factor

    # Apply contrast/brightness/noise, clamp
    # adv_batch = adv_batch * contrast + brightness + noise
    adv_batch = adv_batch + brightness + noise

    # adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
    adv_batch = adv_batch.squeeze()

    return adv_batch




class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.config_file = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.Faster_RCNN = init_detector(self.config_file, self.checkpoint_file, device='cpu').cuda()

        self.darknet_model = Darknet("./models/yolov4.cfg")
        self.darknet_model.load_weights("./models/yolov4.weights")
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        # self.total_variation = TotalVariation().cuda()
        self.mean = torch.Tensor([123.675, 116.28 , 103.53 ]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.Tensor([58.395, 57.12 , 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

        # self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=run_single'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'run_single/{time_str}_{name}')
        else:
            return SummaryWriter()

    def Transform_Patch(self, patch):
        clamp_patch=torch.clamp(patch,0,255)
        unsqueezed_patch = clamp_patch.unsqueeze(0)
        resized_patch = F.interpolate(unsqueezed_patch,(800,800),mode='bilinear').cuda()
        normalized_patch = (resized_patch-self.mean)/self.std
        return normalized_patch

    def Transform_Yolo(self, patch):
        clamp_patch=torch.clamp(patch,0,1)
        unsqueezed_patch = clamp_patch.unsqueeze(0)
        resized_patch = F.interpolate(unsqueezed_patch,(608,608),mode='bilinear').cuda()
        return resized_patch

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 800
        batch_size = 1
        n_epochs = 5000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point

        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")



        # zzj: position set
        patch_position_bias_cpu = torch.full((2, 1), 0)
        patch_position_bias_cpu[0]=0.0
        patch_position_bias_cpu[1]=0.01
        patch_position_bias_cpu.requires_grad_(True)



        # zzj: optimizer = optim.Adam([adv_patch_cpu, patch_position_bias], lr=self.config.start_learning_rate, amsgrad=True)



        et0 = time.time()
        # import csv
        # with open(csv_name, 'w') as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow([0,float(patch_position_bias_cpu[0]), float(patch_position_bias_cpu[1])])


        ####### IMG ########

        img_dir = 'select1000_new'
        super_pixel_dir = 'smaller_outline'
        img_list = os.listdir(img_dir)
        img_list.sort()
        black_img = torch.Tensor(3, 500, 500).fill_(0)
        white_img = torch.Tensor(3, 500, 500).fill_(1)
        white_img_single_layer = torch.Tensor(500, 500).fill_(1)
        black_img_single_layer = torch.Tensor(500, 500).fill_(0)

        img_list =os.listdir('select1000_new')
        
        for img_name in img_list[:]:

            print('------------------------')
            print('------------------------')
            print('Now training', img_name)
            


            ## read image and super-pixel
            img_path = os.path.join(img_dir, img_name)
            super_pixel_path = os.path.join(super_pixel_dir, img_name)

            super_pixel_batch_pil_500 = Image.open(super_pixel_path)
            k = transforms.Compose([transforms.ToTensor()])
            super_pixel_500 = k(super_pixel_batch_pil_500)
            # print(super_pixel_500.shape)
            super_pixel_500_mask = super_pixel_500.repeat(3,1,1)
            super_pixel_500_mask = super_pixel_500_mask.numpy()
            super_pixel_500_mask = super_pixel_500_mask.transpose([1,2,0])

            tf = transforms.Compose([
                transforms.Resize((500,500)),
                transforms.ToTensor()
            ])
            super_pixel_batch = tf(super_pixel_batch_pil_500)
            super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.01), white_img_single_layer, black_img_single_layer)
            if torch.sum(super_pixel_batch_tmp) > 5000:
                super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.1), white_img_single_layer,
                                                    black_img_single_layer)
                if torch.sum(super_pixel_batch_tmp) > 5000:
                    super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.5), white_img_single_layer,
                                                        black_img_single_layer)
            super_pixel_batch = super_pixel_batch_tmp.cuda()

            ## generate patch
            adv_patch_cpu = self.generate_patch("gray")
            adv_patch_cpu.requires_grad_(True)

            img_frcn = get_Image_ready(self.Faster_RCNN,img_path)
            original_result = self.Faster_RCNN(return_loss=False, rescale=True, **img_frcn)
            # original_result = original_result[0]
            original_result = original_result[:,4]
            original_result = original_result[original_result>0.3]
            print("BBoxes predicted: ",original_result.shape[0])

            image_to_patch = mmcv.imread(img_path)
            rgb_image = image_to_patch.copy()
            rgb_image = rgb_image[:,:,::-1].copy()
            img_tensor = torch.Tensor(rgb_image)
            img_tensor = img_tensor.permute([2,0,1])
            # print(img_tensor.shape)
            #500 500 3

            img_path = os.path.join(img_dir, img_name)
            img_clean_pil = Image.open(img_path).convert('RGB')
            img_clean_yolo = transforms.ToTensor()(img_clean_pil)

            resize_small = transforms.Compose([
                transforms.Resize((608, 608)),
            ])
            img0 = resize_small(img_clean_pil)
            boxes2 = do_detect(self.darknet_model, img0, 0.5, 0.4, True)
            print("Yolov4 BBox Predicted: ",len(boxes2))





            # adv_patch_clip_cpu = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu, black_img)

            # optimizer
            optimizer = optim.Adam([
                {'params': adv_patch_cpu, 'lr': self.config.start_learning_rate*1.37}
            ], amsgrad=True)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.3)


            ## img resize
            resize_500 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((500, 500)),
                transforms.ToTensor()
            ])

            super_pixel_mask = super_pixel_batch.repeat(3,1,1).cpu()


            ## rotation start!
            best_so_far =100
            best_conf_so_far = 1.2
            count = 0 

            yolo_below_thresh = False

            for i in tqdm(range(1000)):
                
                adv_patch_frcn_cpu = adv_patch_cpu*256
                new_image = copy.deepcopy(img_frcn)
                patched_img500 = img_tensor*(1-super_pixel_mask)+adv_patch_frcn_cpu*super_pixel_mask
                new_patch = self.Transform_Patch(patched_img500)
                # new_image['img'][0]=new_image['img'][0]*(1-super_pixel_mask)+new_patch*super_pixel_mask
                # new_img['img'][0] = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), new_patch, new_img['img'][0])
                new_image['img'][0]=new_patch

                frcn_output = self.Faster_RCNN(return_loss=False, rescale=True, **new_image)
                # output = output[0]
                frcn_conf = frcn_output[:,4]
                frcn_conf = frcn_conf[frcn_conf>0.26]
                frcn_loss = frcn_conf.sum()

                adv_patch_yolo_cpu = adv_patch_cpu
                patched_yolo = img_clean_yolo*(1-super_pixel_mask)+adv_patch_yolo_cpu*super_pixel_mask
                transformed_yolo = self.Transform_Yolo(patched_yolo)
                yolo_output = self.darknet_model(transformed_yolo)
                # print(len(yolo_output))
                yolo_max_prob = self.prob_extractor(yolo_output)
                # print(yolo_max_prob)

                yolo_loss = yolo_max_prob

                loss = frcn_loss + yolo_loss


                old_patch = adv_patch_cpu.clone()
                if frcn_conf.shape[0]>0 or yolo_loss>0.3:
                    # loss = torch.sum(frcn_conf)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        print()
                        print(float(loss))
                        print("bbox num:",frcn_conf.shape[0])
                        old_patch_np = old_patch.cpu().detach().numpy().transpose([1,2,0])*256
                        old_patch_np = np.round(old_patch_np)
                        old_patch_np = np.clip(old_patch_np,0,255)
                        patched_img = image_to_patch*(1-super_pixel_500_mask)+old_patch_np[:,:,::-1]*super_pixel_500_mask
                        result = inference_detector(self.Faster_RCNN,patched_img)
                        # result = result[0]
                        confidence = result[:,4]
                        confidence = confidence[confidence>0.26]
                        frcn_test = confidence.sum()
                        print("Faster RCNN bbox num:",confidence.shape[0])

                        yolo_result = do_detect(self.darknet_model,transformed_yolo,0.5,0.4,True)
                        yolo_output = self.darknet_model(transformed_yolo)
                        yolo_test = self.prob_extractor(yolo_output)
                        print("Yolo bbox num: ",len(yolo_result))
                        print(yolo_test)

                        test_score = frcn_test+yolo_test

                        if test_score<best_so_far:
                            print("updated!")
                            best_so_far=test_score
                            patched_img=patched_img.astype('int64')
                            
                            diff = np.sum(patched_img-image_to_patch,axis=2)
                            print((diff!=0).sum())


                            cv2.imwrite('smaller_result/'+img_name,patched_img)
                        break


                if i%10==0:
                    with torch.no_grad():
                        print()
                        print(float(loss))
                        print("bbox num:",frcn_conf.shape[0])
                        old_patch_np = old_patch.cpu().detach().numpy().transpose([1,2,0])*256
                        old_patch_np = np.round(old_patch_np)
                        old_patch_np = np.clip(old_patch_np,0,255)
                        patched_img = image_to_patch*(1-super_pixel_500_mask)+old_patch_np[:,:,::-1]*super_pixel_500_mask
                        result = inference_detector(self.Faster_RCNN,patched_img)
                        # result = result[0]
                        confidence = result[:,4]
                        confidence = confidence[confidence>0.26]
                        frcn_test = confidence.sum()
                        print("Faster RCNN bbox num:",confidence.shape[0])

                        yolo_result = do_detect(self.darknet_model,transformed_yolo,0.5,0.4,True)
                        yolo_output = self.darknet_model(transformed_yolo)
                        yolo_test = self.prob_extractor(yolo_output)
                        print("Yolo bbox num: ",len(yolo_result))
                        print(yolo_test)

                        test_score = frcn_test+yolo_test

                        if test_score<best_so_far:
                            print("updated!")
                            best_so_far=test_score
                            patched_img=patched_img.astype('int64')
                            
                            diff = np.sum(patched_img-image_to_patch,axis=2)
                            print((diff!=0).sum())


                            cv2.imwrite('smaller_result/'+img_name,patched_img)
                            if confidence.shape[0]==0 and len(yolo_result)==0:
                                break
                        
                    # elif confidence.shape[0]==best_so_far:
                    #     if confidence.sum()<best_conf_so_far:
                    #         print("updated!")
                    #         best_conf_so_far = confidence.sum()
                    #         patched_img=patched_img.astype('int64')
                        
                    #         diff = np.sum(patched_img-image_to_patch,axis=2)
                    #         print((diff!=0).sum())
                    #         cv2.imwrite('outline_attack/'+img_name,patched_img)


            print("minimum bbox num:",best_so_far)








    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 500, 500), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 500, 500))
        if type == 'trained_patch':
            patchfile = 'patches/object_score.png'
            patch_img = Image.open(patchfile).convert('RGB')
            patch_size = self.config.patch_size
            tf = transforms.Resize((patch_size, patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def connected_domin_detect(input_img):
    from skimage import measure
    # detection
    if input_img.shape[0] ==3:
        input_img_new = (input_img[0]+input_img[1]+input_img[2])
    else:
        input_img_new = input_img
    ones = torch.Tensor(input_img_new.size()).fill_(1)
    zeros = torch.Tensor(input_img_new.size()).fill_(0)
    input_map_new = torch.where((input_img_new != 0), ones, zeros)
    # img = transforms.ToPILImage()(input_map_new.detach().cpu())
    # img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    return float(label_max_number)


def get_obj_min_score(boxes):
    if type(boxes[0][0]) is list:
        min_score_list = []
        for i in range(len(boxes)):
            score_list = []
            for j in range(len(boxes[i])):
                score_list.append(boxes[i][j][4])
            min_score_list.append(min(score_list))
        return np.array(min_score_list)
    else:
        score_list = []
        for j in range(len(boxes)):
            score_list.append(boxes[j][4])
        return np.array(min(score_list))







def main():
    # if len(sys.argv) != 2:
    #     print('You need to supply (only) a configuration mode.')
    #     print('Possible modes are:')
    #     print(patch_config.patch_configs)


    trainer = PatchTrainer("paper_obj")
    trainer.train()

if __name__ == '__main__':
    main()


