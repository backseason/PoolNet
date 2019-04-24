import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self):
        #self.sal_root = './data/DUTS/DUTS-TR'
        #self.sal_source = './data/DUTS/DUTS-TR/train_pair.lst'
        self.sal_root = './data/msrab_hkuis/'
        self.sal_source = './data/msrab_hkuis/msrab_hkuis_train_no_small.lst'

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)


    def __getitem__(self, item):
        # sal data loading
        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item%self.sal_num].split()[0]))
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item%self.sal_num].split()[1]))
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, sal_mode='e'):
        if sal_mode == 'e':
            self.image_root = './data/ECSSD/Imgs/'
            self.image_source = './data/ECSSD/test.lst'
        elif sal_mode == 'p':
            self.image_root = './data/PASCALS/Imgs/'
            self.image_source = './data/PASCALS/test.lst'
        elif sal_mode == 'd':
            self.image_root = './data/DUTOMRON/Imgs/'
            self.image_source = './data/DUTOMRON/test.lst'
        elif sal_mode == 'h':
            self.image_root = './data/HKU-IS/Imgs/'
            self.image_source = './data/HKU-IS/test.lst'
        elif sal_mode == 's':
            self.image_root = './data/SOD/Imgs/'
            self.image_source = './data/SOD/test.lst'
        elif sal_mode == 't':
            self.image_root = './data/DUTS-TE/Imgs/'
            self.image_source = './data/DUTS-TE/test.lst'
        elif sal_mode == 'm_r':
            self.image_root = './data/MSRA/Imgs_resized/'
            self.image_source = './data/MSRA/test_resized.lst'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.image_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item%self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(batch_size, mode='train', num_thread=1, sal_mode='e', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread,
                                      pin_memory=pin)
    else:
        dataset = ImageDataTest(sal_mode=sal_mode)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread,
                                      pin_memory=pin)
    return data_loader

def load_image(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label
