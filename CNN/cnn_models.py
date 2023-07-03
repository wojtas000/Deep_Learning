import torch
import torch.nn as nn
import datasets as ds_own
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.models as models

class ConvolutionalNeuralNetwork():

    """
    Master class for all Convolutional Neural Network models
    being instances of torch.nn.Module class.
    """

    def train_step(self, data, optimizer, criterion):

        """
        Args:
        data: (image, label) tuple.
        optimizer: optimizer chosen for minimization of loss function.
        criterion: loss function.
        Returns:
        Dictionary containing training loss and accuracy. 
        """

        x, y = data

        optimizer.zero_grad()

        logits = self(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}

    
    def test_step(self, data, criterion):

        """
        Args:
        data: (image, label) tuple.
        optimizer: optimizer chosen for minimization of loss function.
        criterion: loss function.
        Returns:
        Dictionary containing testing/validation loss and accuracy. 
        """

        x, y = data

        logits = self(x)
        loss = criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()

        return {'loss': loss, 'accuracy': accuracy}
    
    def Conv2d_output_size(self, w, k, s, p):

        '''
        Method calculating output size of convolutional layer.
        w: width of input image.
        k: kernel size.
        s: stride.
        p: padding.
        Returns:
        Output size of convolutional layer.
        '''

        return np.floor((w - k + 2 * p) / s + 1)
    
    def predict(self, x):

        logits = self(x)
        return torch.softmax(logits, dim=1)
    
    def predict_class(self, x):
        
        logits = self(x)
        return logits.argmax(dim=1)
    
    def prepare_submission(self, test_data=ds_own.cifar_test, dict=ds_own.CLASS_DICT):

        """
        Method used for preparing pandas.DataFrame instance for submission on Kaggle.
        Args:
        test_data: PyTorch Dataset of test data
        dict: dictionary used for encoding labels of data into integers.
        Returns: 
        pandas.DataFrame containing indices of images and their classified labels.
        """
        
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

class Simple_CNN(nn.Module, ConvolutionalNeuralNetwork):

    """
    Class for specific Convolutional Neural Network architecture, having:
    - 2 convolutional layers
    - 2 pooling layers
    - 2 fully connected layers
    - dropout layer
    - activation functions: ReLU
    """

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
        
        super(Simple_CNN, self).__init__()
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


class Complex_CNN(nn.Module, ConvolutionalNeuralNetwork):

    """
    Class for Convolutional Neural Network architecture, coming from https://arxiv.org/ftp/arxiv/papers/2003/2003.13300.pdf
    """

    def __init__(self):

        super(Complex_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=736, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=736, out_channels=508, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=508, out_channels=664, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=664, out_channels=916, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=916, out_channels=186, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=186, out_channels=352, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=22528, out_features=1229)
        self.output = nn.Linear(in_features=1229, out_features=10)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.output(x)
        return x


class PretrainedAlexNet(nn.Module, ConvolutionalNeuralNetwork):

    """
    CLass used for loading pretrained AlexNet model from torchvision.models.
    """

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.model = models.alexnet(pretrained=pretrained)
        
        # Modify the last fully connected layer to output num_classes
        self.model.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
