import os
import librosa
import pandas as pd
import numpy as np
import torch
import pickle
import tensorflow as tf
from torch.utils.data import Dataset
from IPython.display import Audio, display


LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


# Pytorch datasets

class SpeechDataset(Dataset):
    """
    Dataset class for the speech commands dataset, compatible with PyTOrch models.
    """
    def __init__(self, data_dir='train\\audio', labels=LABELS, labels_path='train\\testing_list.txt'):
        """
        Args:
            data_dir (string): Path to the directory containing the audio files.
            labels_path (string): Path to the .txt file containing labels.
            labels (list): List of labels to be used.
        """
        self.data_dir = data_dir
        self.labels_path = labels_path
        self.labels = labels
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.file_paths = []
        self.labels_int = []
        self.sr = 16000
        self.duration = 1 
        
        with open(labels_path, "r") as f:
            lines = f.readlines()
        self.file_paths = [line.strip() for line in lines]
        for file in self.file_paths:
            label = file.split('/')[0]
            if label in self.labels:
                self.labels_int.append(self.labels_dict[label])
            else:
                self.labels_int.append(len(self.labels))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels_int[idx]
        
        audio, sr = librosa.load(os.path.join(self.data_dir, file_path), sr=self.sr, duration=self.duration, mono=True)
        
        audio = torch.FloatTensor(audio)
        
        label = torch.LongTensor([label])
        
        return audio, label
    
    def listen_to_random_sample(self, sample_size=1):
        """
        Listen to a random sample of the dataset.
        Args:
            sample_size (int): Number of samples to listen to.
        """
        sample = np.random.randint(0, len(self), sample_size)
        for idx in sample:
            audio, label = self[idx]
            if label == len(self.labels):
                print('Label: unknown')
            else:
                print('Label:', self.labels[label])
            display(Audio(audio, rate=self.sr))

class ProcessedSpeechDataset(Dataset):
    """
    Dataset class for the preprocessed speech commands dataset (after extracting MFCC features and delta coefficients).
    """
    
    def __init__(self, feature_list='extracted_features\\features_training.pkl', labels=LABELS):
        """
        Args:
            feature_list (string): Path to the .pkl file containing the list of features.
            labels (list): List of labels to be used.
        """
        self.feature_list = feature_list
        self.labels = labels
        self.features = None
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        
        with open(feature_list, "rb") as f:
            self.features = pickle.load(f)

    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        audio = torch.FloatTensor(feature[0])
        
        if feature[1] in self.labels_dict:
            label = torch.LongTensor([self.labels_dict[feature[1]]])
        else:
            label = torch.LongTensor([len(self.labels)])
        
        return audio, label


# Tensorflow datasets


class TensorflowDataset():
    """
    Dataset class for the speech commands dataset, compatible with keras models.
    """
    def __init__(self, pickle_file='extracted_features\\features_validation.pkl', labels=LABELS):
        self.labels = labels
        self.dict = {label: i for i, label in enumerate(self.labels)}
        
        features = pd.read_pickle(pickle_file)
        X = np.array([x[0] for x in features])
        y = np.array([self.dict[x[1]] if x[1] in self.dict.keys() else len(self.labels) for x in features]) 
        
        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))
   
    def __len__(self):
        return len(self.dataset)
   
class TensorflowDataset_unknown():
    """
    Dataset class for the speech commands dataset, compatible with keras models. Used for known vs unknown label detection.
    """
        
    def __init__(self, pickle_file='extracted_features\\features_validation.pkl', labels=LABELS):
        """
        Args:
            pickle_file (string): Path to the .pkl file containing the list of features.
            labels (list): List of labels to be used.
        """
        self.labels = labels
        self.dict = {label: i for i, label in enumerate(self.labels)}
        
        features = pd.read_pickle(pickle_file)
        X = np.array([x[0] for x in features])
        y = np.array([0 if x[1] in self.dict.keys() else 1 for x in features]) 
        
        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    def __len__(self):
        return len(self.dataset)

class TensorflowDataset_labels():
    """
    Dataset class for the speech commands dataset, compatible with keras models. Used for label detection.
    """
     
    def __init__(self, pickle_file='extracted_features\\features_validation.pkl', labels=LABELS):
        """
        Args:
            pickle_file (string): Path to the .pkl file containing the list of features.
            labels (list): List of labels to be used.
        """
        self.labels = labels
        self.dict = {label: i for i, label in enumerate(self.labels)}
        
        features = pd.read_pickle(pickle_file)
        X = np.array([x[0] for x in features])
        y = np.array([self.dict[x[1]] if x[1] in self.dict.keys() else 10 for x in features]) 
        X = X[y != 10]
        y = y[y != 10]
        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def __len__(self):
        return len(self.dataset)


# Datasets

file_dir = os.path.dirname(os.path.abspath(__file__))

label_detection_training = TensorflowDataset(os.path.join(file_dir,'extracted_features\\features_training.pkl'), labels=LABELS).dataset
label_detection_validation = TensorflowDataset(os.path.join(file_dir,'extracted_features\\features_validation.pkl'), labels=LABELS).dataset
label_detection_full = label_detection_training.concatenate(label_detection_validation)

label_detection_training = label_detection_training.shuffle(len(label_detection_training), reshuffle_each_iteration=True)
label_detection_validation = label_detection_validation.shuffle(len(label_detection_validation), reshuffle_each_iteration=True)
label_detection_full = label_detection_full.shuffle(len(label_detection_full), reshuffle_each_iteration=True)

silence_detection_training = TensorflowDataset(os.path.join(file_dir,'extracted_features\\silence_detection_training.pkl'), labels=['silence']).dataset
silence_detection_validation = TensorflowDataset(os.path.join(file_dir,'extracted_features\\silence_detection_validation.pkl'), labels=['silence']).dataset
silence_detection_full = silence_detection_training.concatenate(silence_detection_validation)

silence_detection_training = silence_detection_training.shuffle(len(silence_detection_training), reshuffle_each_iteration=True)
silence_detection_validation = silence_detection_validation.shuffle(len(silence_detection_validation), reshuffle_each_iteration=True)
silence_detection_full = silence_detection_full.shuffle(len(silence_detection_full), reshuffle_each_iteration=True)

unknown_detection_training = TensorflowDataset_unknown(os.path.join(file_dir,'extracted_features\\features_training.pkl'), labels=LABELS).dataset
unknown_detection_validation = TensorflowDataset_unknown(os.path.join(file_dir,'extracted_features\\features_validation.pkl'), labels=LABELS).dataset
unknown_detection_full = unknown_detection_training.concatenate(unknown_detection_validation)

unknown_detection_training = unknown_detection_training.shuffle(len(unknown_detection_training), reshuffle_each_iteration=True)
unknown_detection_validation = unknown_detection_validation.shuffle(len(unknown_detection_validation), reshuffle_each_iteration=True)
unknown_detection_full = unknown_detection_full.shuffle(len(unknown_detection_full), reshuffle_each_iteration=True)

labels_only_detection_training = TensorflowDataset_labels(os.path.join(file_dir,'extracted_features\\features_training.pkl'), labels=LABELS).dataset
labels_only_detection_validation = TensorflowDataset_labels(os.path.join(file_dir,'extracted_features\\features_validation.pkl'), labels=LABELS).dataset
labels_only_detection_full = unknown_detection_training.concatenate(unknown_detection_validation)

labels_only_detection_training = labels_only_detection_training.shuffle(len(labels_only_detection_training), reshuffle_each_iteration=True)
labels_only_detection_validation = labels_only_detection_validation.shuffle(len(labels_only_detection_validation), reshuffle_each_iteration=True)
labels_only_detection_full = labels_only_detection_full.shuffle(len(labels_only_detection_full), reshuffle_each_iteration=True)

