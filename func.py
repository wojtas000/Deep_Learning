import torch
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate

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
