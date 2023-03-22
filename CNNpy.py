import torch
import torch.nn as nn
import torch.optim as optim
import datasets as ds_own
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
import pandas as pd

class ConvolutionalNeuralNetwork():
    
    def train_step(self, data, optimizer, criterion):
        x, y = data

        optimizer.zero_grad()

        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}

    
    def test_step(self, data, criterion):
        x, y = data

        logits = self(x)
        loss = criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}
    
    def Conv2d_output_size(self, w, k, s, p):
        '''
        w - width of input image
        k - kernel size
        s - stride
        p - padding
        '''
        return np.floor((w - k + 2 * p) / s + 1)

class CNN_3_class(nn.Module, ConvolutionalNeuralNetwork):
    def __init__(self, num_classes = 10
                ,kernel_size1=3
                ,kernel_size2=3
                ,stride=1
                ,padding=1
                ,number_of_filters0=32
                ,number_of_filters1=256
                ,length_of_input0=32
                ,no_neurons = 500
                ,dr=nn.Dropout(p=0)
                ,activation_function=torch.relu):
        super(CNN_3_class, self).__init__()
        self.conv1 = nn.Conv2d(3, number_of_filters0, kernel_size1, stride, padding)
        self.pool1 = nn.MaxPool2d(2)
        length0 = self.Conv2d_output_size(length_of_input0, kernel_size1, stride, padding)//2
        self.conv2 = nn.Conv2d(number_of_filters0, number_of_filters1, kernel_size2, stride, padding)
        self.pool2 = nn.MaxPool2d(2)
        length1 = self.Conv2d_output_size(length0, kernel_size2, stride, padding)//2
        self.fc1 = nn.Linear(int(length1*length1*number_of_filters1), no_neurons)
        self.fc2 = nn.Linear(no_neurons, num_classes)
        self.dr = dr
        self.activation_function = activation_function

    def forward(self, x):
        x = self.pool1(self.activation_function(self.conv1(x)))
        x = self.pool2(self.activation_function(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.activation_function(self.fc1(x))
        x = self.dr(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        logits = self(x)
        return torch.softmax(logits, dim=1)
    
    def predict_class(self, x):
        logits = self(x)
        return logits.argmax(dim=1)
    
    def prepare_submission(self, test_data=ds_own.cifar_test, dict=ds_own.CLASS_DICT):
        
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        all_predictions = []
        
        with torch.no_grad():
            print('Classifying test images...')
            for images, _ in tqdm(test_loader):
                predicted = self.predict_class(images)
                all_predictions += predicted.tolist()
        
        r_dict = {value:key for key, value in dict.items()}

        labels = [r_dict[pred] for pred in all_predictions]
        id = list(range(1, len(all_predictions) + 1))
        submission = pd.DataFrame({'id': id, 'label': labels})
        
        return submission

# class CNN_3_class(nn.Module, ConvolutionalNeuralNetwork):
#     def __init__(self, num_classes = 10
#                 ,kernel_size1=3
#                 ,kernel_size2=3
#                 ,stride=1
#                 ,padding=1
#                 ,number_of_filters0=32
#                 ,number_of_filters1=32
#                 ,length_of_input0=32
#                 ,no_neurons = 128
#                 ,dr=nn.Dropout(p=0)
#                 ,activation_function=torch.relu):
#         super(CNN_3_class, self).__init__()
#         self.conv1 = nn.Conv2d(3, number_of_filters0, kernel_size1, stride, padding, stride)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(number_of_filters0, number_of_filters1, kernel_size2, stride, padding, stride)
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(number_of_filters1 * (length_of_input0//4) * (length_of_input0//4), no_neurons)
#         self.fc2 = nn.Linear(no_neurons, num_classes)
#         self.dr = dr
#         self.activation_function = activation_function

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.activation_function(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.activation_function(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.activation_function(x)
#         x = self.dr(x)
#         x = self.fc2(x)
#         return x

# class CNN_3_class(nn.Module, ConvolutionalNeuralNetwork):
#     def __init__(self, num_classes = 10
#                 ,kernel_size1=2
#                 ,kernel_size2=2
#                 ,stride=1
#                 ,padding=1
#                 ,number_of_filters0=32
#                 ,number_of_filters1=32
#                 ,length_of_input0=32
#                 ,no_neurons = 128
#                 ,dr=nn.Dropout(p=0)
#                 ,activation_function=torch.relu):
#         super(CNN_3_class, self).__init__()
#         self.conv1 = nn.Conv2d(3, number_of_filters0, kernel_size1, stride, padding)
#         self.pool1 = nn.MaxPool2d(2)
#         length_of_input1 = self.Conv2d_output_size(length_of_input0, kernel_size1, stride, padding)//2
#         self.conv2 = nn.Conv2d(number_of_filters0, number_of_filters1, kernel_size2, stride, padding)
#         self.pool2 = nn.MaxPool2d(2)
#         length_of_input2 = self.Conv2d_output_size(length_of_input1, kernel_size2, stride, padding)//2
#         self.fc1 = nn.Linear(int(number_of_filters1*length_of_input2*length_of_input2), no_neurons)
#         self.dr = dr
#         self.fc2 = nn.Linear(no_neurons, num_classes)
#         # parameters
#         self.num_classes = num_classes
#         self.kernel_size1 = kernel_size1
#         self.kernel_size2 = kernel_size1# change into the same size as kernel_size1
#         self.stride = stride
#         self.padding = padding
#         self.number_of_filters0 = number_of_filters0
#         self.number_of_filters1 = number_of_filters1
#         self.length_of_input0 = length_of_input0
#         self.no_neurons = no_neurons

#         self.activation_function = activation_function
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.activation_function(x)
#         x = self.pool1(x)
#         length_of_input1 = self.Conv2d_output_size(self.length_of_input0, self.kernel_size1, self.stride, self.padding)/2
#         x = self.conv2(x)
#         x = self.activation_function(x)
#         x = self.pool2(x)
#         length_of_input2 = self.Conv2d_output_size(length_of_input1, self.kernel_size2, self.stride, self.padding)/2
#         x = x.view(-1, int(self.number_of_filters1*length_of_input2*length_of_input2))
#         x = self.fc1(x)
#         x = self.activation_function(x)
#         x = self.dr(x)
#         x = self.fc2(x)
#         return x

import torch.nn as nn
import torchvision.models as models

class PretrainedAlexNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.model = models.alexnet(pretrained=pretrained)
        
        # Modify the last fully connected layer to output num_classes
        self.model.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x