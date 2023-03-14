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


# Path to training and validation directories
TRAIN_DIR = 'Cifar10\\train'
VAL_DIR = 'Cifar10\\val'

# Path to dataframe with labels for training and validation data
TRAIN_LABELS = 'Cifar10\\trainLabels.csv'
VAL_LABELS = 'Cifar10\\valLabels.csv'

# List of class names
CLASS_NAMES = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog',
 'airplane']

# Dictionary for encoding class names
CLASS_DICT = {CLASS_NAMES[i]: i for i in range(len(CLASS_NAMES))}

# Size in pixels of single image
IMG_SIZE=32


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


class MixedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.permutation = list(range(len(self.dataset1)))
        random.shuffle(self.permutation)
    
    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        # get images from the two Datasets
        image1, label = self.dataset1[index]
        image2, _ = self.dataset2[self.permutation[index]]

        # mix the two images using tensor_mixer
        mixed_image = MixedDataset.tensor_mixer(image1, image2)

        # return the mixed image and a dummy label
        return mixed_image, label
    
    @staticmethod
    def tensor_mixer(tensor1, tensor2):
        return torch.mean(torch.stack([tensor1, tensor2]), dim=0)

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
            label = self.dataset1.labels.iloc[i, 1]
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



# Create pyTorch Dataset instances of training and validation data
cifar_train = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                           transform=transforms.ToTensor(), class_dict=CLASS_DICT)
cifar_val = CifarDataset(root_dir = VAL_DIR, labels=VAL_LABELS, 
                         transform=transforms.ToTensor(), class_dict=CLASS_DICT)

# Simple augmentation technique
transformer1 = transforms.Compose([
                                  transforms.RandomRotation(degrees=(-30,30)),
                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                  transforms.RandomCrop(int(IMG_SIZE * 0.7)),
                                  transforms.ToTensor(),
                                  GaussianNoise(mean=0, std=0.01)])
                                    


cifar_aug1 = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                          transform=transformer1, class_dict=CLASS_DICT)

# Merge training data with augmented data
merged_dataset1 = ConcatDataset([cifar_train, cifar_aug1])

# Advanced augmentation technique - mixing
transformer2 = transforms.Compose([transforms.Pad(padding=(4, 4, 4, 4), fill=0, padding_mode='constant'),
                                  transforms.RandomCrop(IMG_SIZE),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.ToTensor()])
                                  
                                    
cifar_aug2 = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                                      transform=transformer2, class_dict=CLASS_DICT), 
cifar_aug3 = CifarDataset(root_dir = TRAIN_DIR, labels=TRAIN_LABELS, 
                          transform=transformer2, class_dict=CLASS_DICT)

mixed_dataset = MixedDataset(cifar_aug2, cifar_aug3)

# Merge training data with mixed data
merged_dataset2 = ConcatDataset([cifar_train, mixed_dataset])

# Create DataLoader instances
train_loader = DataLoader(cifar_train, batch_size=32, shuffle=True)
val_loader = DataLoader(cifar_val, batch_size=32, shuffle=False)
augmentation1_loader = DataLoader(merged_dataset1, batch_size=32, shuffle=True)
augmentation2_loader = DataLoader(merged_dataset2, batch_size=32, shuffle=True)