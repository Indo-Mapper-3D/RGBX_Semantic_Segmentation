import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
import cv2


img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'

image_w = 640
image_h = 480

class customDataset(Dataset):
    def __init__(self,img_dir,dep_dir):


       self.img_dir = img_dir
       self.dep_dir = dep_dir
       
       img_list = os.listdir(img_dir)
       self.scans_list = list(map(lambda x: x.split('.')[0],img_list))



    def __len__(self):
        return len(self.scans_list)

    def __getitem__(self, idx):
        
        img_path = self.img_dir + '/' + self.scans_list[idx]+'.png'
        dep_path = self.dep_dir + '/' + self.scans_list[idx]+'.npy'


        image = np.array(imageio.imread(img_path))
        depth = np.load(dep_path)
        depth = cv2.merge([depth,depth,depth])

        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                        mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                        mode='reflect', preserve_range=True)

        image = image.astype(np.float32)
        depth = depth.astype(np.float32)

        # image = image / 255
        # image = torch.from_numpy(image).float()
        # depth = torch.from_numpy(depth).float()
        # image = image.permute(2, 0, 1)
        # depth.unsqueeze_(0)

        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225])(image)
        # depth = torchvision.transforms.Normalize(mean=[19050],
        #                                         std=[9650])(depth)
        
        return {'image' : image, 'depth' : depth , 'name' : self.scans_list[idx]}
