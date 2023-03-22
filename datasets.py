"""
This module implements pyTorch Dataset custom classes (for loading Cifar10 data) and creates 
instances of pyTorch Datasets and DataLoaders of training and validation data  
"""

import torch
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.nn.functional import interpolate
import torchvision.transforms.functional as TF


# GLOBAL VARIABLES

# Path to training and validation directories
TRAIN_DIR = 'Cifar10\\train'
VAL_DIR = 'Cifar10\\val'
TRAIN_FULL_DIR = 'Cifar10\\train_full'
TEST_DIR = 'Cifar10\\test'

# Path to dataframe with labels for training and validation data
TRAIN_LABELS = 'Cifar10\\trainLabels.csv'
VAL_LABELS = 'Cifar10\\valLabels.csv'
TRAIN_FULL_LABELS = 'Cifar10\\train_fullLabels.csv'
TEST_LABELS = 'Cifar10\\testLabels.csv'

# List of class names
CLASS_NAMES = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog',
 'airplane']

# Dictionary for encoding class names
CLASS_DICT = {CLASS_NAMES[i]: i for i in range(len(CLASS_NAMES))}

# Size in pixels of single image
IMG_SIZE=32


# CLASSES

class CifarDataset(Dataset):
    """
    Class for storing dataset used for feeding Neural Network. 
    """
    
    def __init__(self, root_dir, labels, class_dict, transform=None):
        self.root_dir = root_dir
        self.labels = pd.read_csv(labels, header=0)
        self.transform = transform
        self.class_dict = class_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_id = self.labels.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, str(img_id) + '.png'))
        label = self.class_dict[self.labels.iloc[index, 1]]

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)
    
    def display_image(self, index=0, resize=128):
        """
        Function for displaying single image from dataset, indexed by 'index' parameter.
        Params:
        index - index of image
        resize - value used for resizing the image. For example if resize=128, we get 128x128 pixels image.
        """
        img, _ = self[index]
        label = self.labels.iloc[index, 1]
        resized_img = interpolate(img.unsqueeze(0), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(0)
        plt.title(label)
        plt.imshow(resized_img.permute(1,2,0))
        
    def display_sample_images(self, sample=None, sample_size=4, resize=128):
        """
        Function for displaying sample of images from dataset. The sample can be random or chosen by the user.
        Params:
        sample - list of indices of the images in Dataset
        sample_size - number of randomly selected images, if sample == None
        resize - value used for resizing the image. For example if resize=128, we get 128x128 pixels image.
        """
        if sample:
            random_sample = sample
        else:
            random_sample = random.sample(range(len(self)), sample_size)
        fig, axes = plt.subplots(nrows = int(np.ceil(sample_size/2)) , ncols = 2, figsize=(6, sample_size))
        row, col = 0, 0
        
    
        for i in random_sample:
            img, _ = self[i]
            label = self.labels.iloc[i, 1]
            resized_img = interpolate(img.unsqueeze(0), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(0)

            if sample_size <=2:
                axes[col].imshow(resized_img.permute(1,2,0))
                axes[col].set_title(label)
            else:
                axes[row, col].imshow(resized_img.permute(1,2,0))
                axes[row, col].set_title(label)
            
            if col == 0:
                col += 1
            else:
                row += 1
                col = 0
        
        fig.suptitle('Random sample')
        fig.subplots_adjust(hspace=0.35)
        plt.show()

class GaussianNoise(object):
    """
    Class used for adding Gaussian noise to image. 
    Use after transforms.ToTensor(), if used in transforms.Compose() pipeline. 
    """
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, img):
        return img + torch.randn(tuple(img.size())) * self.std + self.mean


class Cutout(object):
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        mask = np.ones((h, w), np.float32)
        mask[y:y+self.size, x:x+self.size] = 0.
        mask = torch.from_numpy(mask)
        img = TF.to_tensor(img)
        img = img * mask.unsqueeze(0)
        img = TF.to_pil_image(img)

        return img

class MixUpTransform(object):
    def __init__(self, alpha=0.5, dataset = None):
        self.alpha = alpha
        self.dataset = dataset

    def __call__(self, img):
        mix_img, _ = self.dataset[int(torch.randperm(len(self.dataset))[0])]

        mixed = img * self.alpha + mix_img * (1 - self.alpha) 

        return mixed

# ORIGINAL TRAIN AND VALIDATION DATASETS

default_transformer = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,))])

cifar_train = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                           transform=default_transformer, class_dict=CLASS_DICT)
cifar_val = CifarDataset(root_dir = VAL_DIR, labels=VAL_LABELS, 
                         transform=default_transformer, class_dict=CLASS_DICT)
cifar_test = CifarDataset(root_dir = TEST_DIR, labels=TEST_LABELS, 
                           transform=default_transformer, class_dict={0.0:0})

# Simple augmentation technique
transformer1 = transforms.Compose([transforms.Pad(padding=(4, 4, 4, 4), fill=0, padding_mode='constant'),
                                  transforms.RandomRotation(degrees=(-30,30)),
                                  transforms.ColorJitter(brightness=0.4),
                                  transforms.RandomCrop(IMG_SIZE),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  GaussianNoise(mean=0, std=0.01)])


# Advanced augmentation technique - mixing
transformer2 = transforms.Compose([transforms.Pad(padding=(4, 4, 4, 4), fill=0, padding_mode='constant'),
                                  transforms.RandomCrop(IMG_SIZE),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  MixUpTransform(alpha=0.7, dataset=cifar_train)])
                                  

# Advanced augmentation technique - cutout
transformer3 = transforms.Compose([Cutout(size=10, p=1),
                                  transforms.ToTensor(), 
                                  transforms.Normalize((0.5,), (0.5,))])
                                  
                                    
# DATASETS, DATALOADERS

# Original
cifar_train = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                           transform=default_transformer, class_dict=CLASS_DICT)
cifar_train_full = CifarDataset(root_dir = TRAIN_FULL_DIR, labels=TRAIN_FULL_LABELS, 
                           transform=default_transformer, class_dict=CLASS_DICT)

cifar_val = CifarDataset(root_dir = VAL_DIR, labels=VAL_LABELS, 
                         transform=default_transformer, class_dict=CLASS_DICT)

train_loader = DataLoader(cifar_train, batch_size=32, shuffle=True)
val_loader = DataLoader(cifar_val, batch_size=32, shuffle=False)

#augmented
cifar_basic_aug = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                          transform=transformer1, class_dict=CLASS_DICT)
cifar_basic_aug_full = CifarDataset(root_dir = TRAIN_FULL_DIR, labels=TRAIN_FULL_LABELS, 
                          transform=transformer1, class_dict=CLASS_DICT)

cifar_mixup = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                                      transform=transformer2, class_dict=CLASS_DICT)
cifar_cutout = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                          transform=transformer3, class_dict=CLASS_DICT)
basic_aug_loader = DataLoader(cifar_basic_aug, batch_size=32, shuffle=True)
mixup_loader = DataLoader(cifar_mixup, batch_size=32, shuffle=True)
cutout_loader = DataLoader(cifar_cutout, batch_size=32, shuffle=True)


# dataset to alexnet

transformer = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Create pyTorch Dataset instances of training and validation data
cifar_train_AN = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                           transform=transformer, class_dict=CLASS_DICT)
cifar_val_AN = CifarDataset(root_dir = VAL_DIR, labels=VAL_LABELS, 
                         transform=transformer, class_dict=CLASS_DICT)

cifar_train_fullAN = CifarDataset(root_dir = TRAIN_FULL_DIR, labels=TRAIN_FULL_LABELS,
                            transform=transformer, class_dict=CLASS_DICT)

train_loader_AN = DataLoader(cifar_train_AN, batch_size=32, shuffle=True)
test_loader_AN = DataLoader(cifar_val_AN, batch_size=32, shuffle=False)
