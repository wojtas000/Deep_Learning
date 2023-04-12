import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython.display import Audio, display

LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
LABELS_DICT = {label: i for i, label in enumerate(LABELS)}


class SpeechDataset(Dataset):
    
    def __init__(self, data_dir, labels=LABELS):
        self.data_dir = data_dir
        self.labels = labels
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.file_paths = []
        self.labels_int = []
        self.sr = 16000
        self.duration = 1 
        for dir in os.listdir(data_dir):
            if dir in self.labels:
                for file in os.listdir(os.path.join(data_dir, dir)):
                    self.file_paths.append(os.path.join(data_dir, dir, file))
                    self.labels_int.append(self.labels_dict[dir])
            elif dir != '_background_noise_':
                for file in os.listdir(os.path.join(data_dir, dir)):
                    self.file_paths.append(os.path.join(data_dir, dir, file))
                    self.labels_int.append(len(self.labels))
            else:
                pass
            
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels_int[idx]
        
        audio, sr = librosa.load(file_path, sr=self.sr, duration=self.duration, mono=True)
        
        audio = torch.FloatTensor(audio)
        
        label = torch.LongTensor([label])
        
        return audio, label
    
    def listen_to_random_sample(self, sample_size=1):
        sample = np.random.randint(0, len(self), sample_size)
        for idx in sample:
            audio, label = self[idx]
            print('Label:', self.labels[label])
            display(Audio(audio, rate=self.sr))