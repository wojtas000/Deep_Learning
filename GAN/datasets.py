import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np

from torch.utils.data import TensorDataset


dataset_path = "D:\\deep\\data0\\lsun\\bedroom"
#dataset_path = 'lsun\\bedroom'

# Funkcja do odczytywania zdjęć z folderu
def load_images_from_folder(folder_path):
    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
    return images

def reshape_matrices(array):
    size_x = 256
    size_y = 256
    size_z = 3

    num_matrices = array.shape[0]
    reshaped_array = np.zeros((num_matrices, size_x, size_y, size_z))

    for i in range(num_matrices):
        reshaped_array[i] = np.resize(array[i], (size_x, size_y, size_z))

    # transpoze the last dimension to match the pytorch format
    reshaped_array = np.transpose(reshaped_array, (0, 3, 1, 2))

    return reshaped_array

# Przykładowe użycie
images = []
for folder1 in [0]:
    for folder2 in [0]:
        for folder3 in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']:
            print(os.path.join(dataset_path, str(folder1), str(folder2), str(folder3)))
            current_folder_path = os.path.join(dataset_path, str(folder1), str(folder2), str(folder3))
            images += load_images_from_folder(current_folder_path)

# preprocessing by resizing images to 64x64 and normalizing them
images = np.array(images)
images = images / 255.
images = reshape_matrices(images)

# convert images to tensors and torch.float32

images = torch.from_numpy(images)
images = images.type(torch.float32)

# Prepare Dataloader
y = torch.ones(images.shape[0])
dataset = TensorDataset(images, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)