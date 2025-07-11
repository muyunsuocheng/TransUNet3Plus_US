import torch
import torch.nn as nn
import os

import torchvision.transforms
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

import json
import cv2
import numpy as np

class SingleClassDataset(torch.utils.data.Dataset) :
    """loads input images and binary masks from a root directory, returns a tuple containing
      the input image, the mask, and a 1-hot encoded vector indicating if the image is a negative or positive.

      prerequesites :
        -the structure of the root directory is  root
                                                   '-->Images
                                                   '-->Masks
        -every mask file and image file has the same extension
        """
    def __init__(self,root_directory,
                 image_shape = [256,256],  # 改为256x256
                 image_ext = ".png",
                 mask_ext= ".png",
                 _device = 0):

        super().__init__()
        im_dir = os.path.join(os.path.dirname(root_directory), 'data', 'train_set','Images') + '/'
        mask_dir = os.path.join(os.path.dirname(root_directory), 'data', 'train_set','Masks') + '/'
        self.image_list = [im_dir+f for f in sorted(os.listdir(im_dir)) if f.endswith(image_ext)]
        self.mask_list = [f.replace(im_dir,mask_dir).replace(image_ext,mask_ext) for f in self.image_list]

        # 双线性插值resize图片，mask用最近邻插值
        self.resizer = torchvision.transforms.Resize(image_shape, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.resizer_mask = torchvision.transforms.Resize(image_shape, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.device = torch.device(_device) if _device >= 0  else torch.device("cpu")

    def __getitem__(self, item):
        # 读取并归一化图片
        im = read_image(self.image_list[item]).float()
        im /= im.max()
        mask = read_image(self.mask_list[item]).float()

        # 如果mask是三通道，取第一个通道
        if mask.shape[0] > 1:
            mask = mask[0:1, :, :]

        one_hot_mask = torch.zeros([2, *mask.shape[1:]], dtype=torch.float)
        one_hot_mask[1, :] = (mask > 0.5).float()
        one_hot_mask[0, :] = (mask <= 0.5).float()
        mask = one_hot_mask

        im = self.resizer(im)
        mask = self.resizer_mask(mask)

        # 随机增强
        if torch.randint(1, 100, [1]).item() < 50:
            im = TF.hflip(im)
            mask = TF.hflip(mask)
        if torch.randint(1, 100, [1]).item() < 50:
            im = TF.vflip(im)
            mask = TF.vflip(mask)

        pres = (torch.count_nonzero(mask) != 0).float().item()
        one_hot_presence = torch.Tensor([1 - pres, pres])

        im = im.to(self.device)
        mask = mask.to(self.device)
        one_hot_presence = one_hot_presence.to(self.device)

        return im, mask, one_hot_presence

    def __len__(self):
        return len(self.image_list)

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, rootdir=None, image_dir=None, mask_dir=None,
                 image_shape=[256, 256], _device=0):
        super().__init__()
        self.resizer = T.Resize(image_shape, interpolation=T.InterpolationMode.BILINEAR)
        self.resizer_mask = T.Resize(image_shape, interpolation=T.InterpolationMode.NEAREST)
        if image_dir is not None:
            self.image_list = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            if mask_dir is not None and os.path.exists(mask_dir):
                self.mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
            else:
                self.mask_list = None  # 测试集没有mask
            self.categories = [0, 1]  # 默认二分类
        elif rootdir is not None:
            with open(rootdir + '_annotations.coco.json') as fopen:
                Parsed = json.loads(fopen.read())
            self.root = rootdir
            self.image_list = [fpath for fpath in Parsed["images"]]
            self.categories = Parsed["categories"]
            self.annotations = Parsed["annotations"]
        else:
            raise ValueError("COCODataset: 必须指定 rootdir 或 image_dir 和 mask_dir。")

    def __getitem__(self, item):
        if hasattr(self, 'mask_list') and self.mask_list is not None:
            # 训练/验证集：返回图片和掩码
            im = read_image(self.image_list[item]).float()
            im /= im.max()
            mask = read_image(self.mask_list[item]).float()
            if mask.shape[0] > 1:
                mask = mask[0:1, :, :]
            one_hot_mask = torch.zeros([2, *mask.shape[1:]], dtype=torch.float)
            one_hot_mask[1, :] = (mask > 0.5).float()
            one_hot_mask[0, :] = (mask <= 0.5).float()
            mask = one_hot_mask
            im = self.resizer(im)
            mask = self.resizer(mask)
            return im, mask
        elif hasattr(self, 'mask_list') and self.mask_list is None:
            # 测试集：只返回图片 
            img = read_image(self.image_list[item]).float()
            img /= img.max()  # 与训练时一致
            im = self.resizer(img)
            return im
        else:
            # COCO方式（如有需要可保留，否则可删）
            im = read_image(self.root + self.image_list[item]["file_name"]).float()
            im /= im.max()
            one_hot_mask = np.zeros([len(self.categories), *im.shape[-2:]], dtype=np.uint8)
            one_hot_mask[0, ...] = 1
            image_annotations = [annot_dict for annot_dict in self.annotations if annot_dict['image_id'] == self.image_list[item]['id']]
            for annotation in image_annotations:
                cls_index = annotation["category_id"]
                polypoints = np.reshape(annotation["segmentation"][0], (len(annotation["segmentation"][0]) // 2, 2)).astype(np.int32)
                one_hot_mask[cls_index, ...] = cv2.fillPoly(one_hot_mask[cls_index, ...], [polypoints], 1)
                one_hot_mask[0, ...] = cv2.fillPoly(one_hot_mask[0, ...], [polypoints], 0)
            one_hot_mask = torch.Tensor(one_hot_mask).float()
            im = self.resizer(im)
            one_hot_mask = self.resizer(one_hot_mask)
            return im, one_hot_mask

    def __len__(self):
        return len(self.image_list)
