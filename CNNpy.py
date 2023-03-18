import torch
import torch.nn as nn
import torch.optim as optim
import datasets as ds_own
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np

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
                ,kernel_size1=2
                ,kernel_size2=2
                ,stride=1
                ,padding=1
                ,number_of_filters0=32
                ,number_of_filters1=32
                ,length_of_input0=32
                ,no_neurons = 128
                ,dr=nn.Dropout(p=0)
                ,activation_function=torch.relu):
        super(CNN_3_class, self).__init__()
        self.conv1 = nn.Conv2d(3, number_of_filters0, kernel_size1, stride, padding)
        self.pool1 = nn.MaxPool2d(2)
        length_of_input1 = self.Conv2d_output_size(length_of_input0, kernel_size1, stride, padding)//2
        self.conv2 = nn.Conv2d(number_of_filters0, number_of_filters1, kernel_size2, stride, padding)
        self.pool2 = nn.MaxPool2d(2)
        length_of_input2 = self.Conv2d_output_size(length_of_input1, kernel_size2, stride, padding)//2
        self.fc1 = nn.Linear(int(number_of_filters1*length_of_input2*length_of_input2), no_neurons)
        self.dr = dr
        self.fc2 = nn.Linear(no_neurons, num_classes)
        # parameters
        self.num_classes = num_classes
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size1# change into the same size as kernel_size1
        self.stride = stride
        self.padding = padding
        self.number_of_filters0 = number_of_filters0
        self.number_of_filters1 = number_of_filters1
        self.length_of_input0 = length_of_input0
        self.no_neurons = no_neurons

        self.activation_function = activation_function
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.pool1(x)
        length_of_input1 = self.Conv2d_output_size(self.length_of_input0, self.kernel_size1, self.stride, self.padding)/2
        x = self.conv2(x)
        x = self.activation_function(x)
        x = self.pool2(x)
        length_of_input2 = self.Conv2d_output_size(length_of_input1, self.kernel_size2, self.stride, self.padding)/2
        x = x.view(-1, int(self.number_of_filters1*length_of_input2*length_of_input2))
        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.dr(x)
        x = self.fc2(x)
        return x