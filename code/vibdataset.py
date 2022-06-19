import pandas as pd
from pathlib import Path
import torch
import csv
import numpy as np
import random
from torchaudio import transforms
import matplotlib.pyplot as plt
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import librosa
import torch.onnx
import pywt
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import cv2
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
import seaborn
from torchsummary import summary
from torch.autograd import Variable
from skimage import restoration
from tool import VibrationSignal


class VibrationDS2(Dataset):
    # return [3,401,480] spec matrix
    def __init__(self, metadata_path, data_path, channel, if_wt, if_spec):
        # metadata_path: path to metadata file; data_path: path to the folder that contains folder
        # channel: 1 or 3
        # if_wt: whether to use wavelet transform
        # if_spec whether to transform to spectrogram if it's true channel must be 3
        self.metadata_path = metadata_path
        self.data_path = data_path
        df = pd.read_csv(metadata_path)
        df['relative_path'] = '/' + df['folder'].astype(str) + '/' + df['filename'].astype(str)
        self.df_n = df[['relative_path', 'class']]
        self.sample_rate = 500
        self.channel = channel
        self.if_wt = if_wt
        self.if_spec = if_spec

    def __len__(self):
        return len(self.df_n)

    def __getitem__(self, idx):
        vibration_file = self.data_path + self.df_n.loc[idx, 'relative_path']
        class_id = self.df_n.loc[idx, 'class']
        if self.if_wt:
            vib0_0 = VibrationSignal.get_vib(vibration_file, self.channel)
            vib0_1 = VibrationSignal.time_stretch(0.5, 2, vib0_0)
            vib0_1 = VibrationSignal.vibration_wavelet7(vib0_1, self.channel, wavelet_type='db10')
        else:
            vib0_1 = VibrationSignal.get_vib(vibration_file, self.channel)
            vib0_1 = VibrationSignal.time_stretch(1, 1, vib0_1)
        vib0 = VibrationSignal.vibration2tensor(vib0_1, self.channel)
        vib1 = VibrationSignal.padding2(vib0, self.channel, max_point=632)  # 632
        # vib1 = vib0
        vib2, f = VibrationSignal.time_shift(vib1, shift_limit=0.6)
        vib = vib2
        if self.if_spec:
            vib = VibrationSignal.spectrogram(vib2, n_fft=150, win_len=20)
            vib = transforms.AmplitudeToDB()(vib)
            for i in range(self.channel):
                m1, m2 = vib[i].min(), vib[i].max()
                vib[i] = (vib[i] - m1) / (m2 - m1)
                vib[i] = nn.Threshold(0.4, 0)(vib[i])
            # vib = VibrationSignal.spectro_augment(vib)

        return vib, class_id


class VibrationDS(Dataset):
    # return [3,401,480] spec matrix
    def __init__(self, metadata_path, data_path, channel, if_wt, if_spec):
        # metadata_path: path to metadata file; data_path: path to the folder that contains folder
        # channel: 1 or 3
        # if_wt: whether to use wavelet transform
        # if_spec whether to transform to spectrogram if it's true channel must be 3
        self.metadata_path = metadata_path
        self.data_path = data_path
        df = pd.read_csv(metadata_path)
        df['relative_path'] = '/' + df['folder'].astype(str) + '/' + df['filename'].astype(str)
        self.df_n = df[['relative_path', 'class']]
        self.sample_rate = 500
        self.channel = channel
        self.if_wt = if_wt
        self.if_spec = if_spec

    def __len__(self):
        return len(self.df_n)

    def __getitem__(self, idx):
        vibration_file = self.data_path + self.df_n.loc[idx, 'relative_path']
        class_id = self.df_n.loc[idx, 'class']
        if self.if_wt:
            vib0_0 = VibrationSignal.get_vib(vibration_file, self.channel)
            vib0_1 = VibrationSignal.time_stretch(1.3, 1.3, vib0_0)
            vib0_1 = VibrationSignal.vibration_wavelet2(vib0_1, self.channel, wavelet_type='db10')
        else:
            vib0_1 = VibrationSignal.get_vib(vibration_file, self.channel)
            vib0_1 = VibrationSignal.time_stretch(1, 1, vib0_1)
        vib0 = VibrationSignal.vibration2tensor(vib0_1, self.channel)
        vib1 = VibrationSignal.padding2(vib0, self.channel, max_point=632)
        # vib1 = vib0
        # vib2, f = VibrationSignal.time_shift(vib1, shift_limit=0.6)
        vib = vib1
        if self.if_spec:
            vib = VibrationSignal.spectrogram(vib2, n_fft=150, win_len=20)
            vib = transforms.AmplitudeToDB()(vib)
            for i in range(self.channel):
                m1, m2 = vib[i].min(), vib[i].max()
                vib[i] = (vib[i] - m1) / (m2 - m1)
                vib[i] = nn.Threshold(0.4, 0)(vib[i])
            # vib = VibrationSignal.spectro_augment(vib)

        return vib, class_id
