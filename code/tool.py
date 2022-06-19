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


class VibrationSignal:
    # signal processing method for cnn and dnn
    # the last dimension of the tensor should be the length of the vib signal
    @staticmethod
    def get_vib(data_path, channel):
        # if channel = 3 return [sensor1,sensor3,sensor5]
        # if channel = 1 return [sensor1]
        vibration_raw = pd.read_csv(data_path, header=None)
        if channel == 3:
            data = [vibration_raw[1].values, vibration_raw[3].values, vibration_raw[5].values]
        elif channel == 1:
            data = [vibration_raw[1].values]
        return data

    @staticmethod
    def vibration_wavelet(vib, channel, wavelet_type='coif5', threshold=0.2):
        # return a list of reconstructed signal [sig_re] or [sig1_re,sig3_re,sig5_re]
        # sig_re is a list
        wavelet_type = wavelet_type
        channel = channel
        vib = vib
        threshold = threshold
        vib_re = []
        for d in vib:
            coe = pywt.wavedec(d, wavelet_type)
            for i in range(1, len(coe)):
                coe[i] = pywt.threshold(coe[i], threshold * max(coe[i]))
            vib_re.append(pywt.waverec(coe, wavelet_type))
        return vib_re

    @staticmethod
    def vibration_wavelet2(vib, channel, wavelet_type='rbio2.6'):
        wavelet_type = wavelet_type
        channel = channel
        vib = vib
        vib_re = []
        for d in vib:
            coe = pywt.wavedec(d, wavelet_type)
            vib_re.append(pywt.waverec(coe[0:4], wavelet_type))
        return vib_re

    @staticmethod
    def vibration_wavelet7(vib, channel, wavelet_type='db10'):
        vib_re = []
        for d in vib:
            coe = pywt.wavedec(d, wavelet_type)
            for i in range(1, len(coe)):
                T = restoration.estimate_sigma(coe[i])
                coe[i] = pywt.threshold(coe[i], T, mode='hard')
            vib_re.append(pywt.waverec(coe[0:4], wavelet_type))
        return vib_re

    @staticmethod
    def time_stretch(l, h, vib):
        # input form [sensor] or [sensor1, sensor3, sensor5], where sensor: list or array
        # follwing get_vib()
        # return [sensor] or [sensor1, sensor3, sensor5]
        stretch_para = random.uniform(l, h)
        # stretch_para=2 # test
        re_vib = []
        for d in vib:
            d = d.astype('float32')
            leng = int(len(d) * stretch_para)
            re_vib.append(cv2.resize(d, (1, leng)).squeeze())
        return re_vib

    @staticmethod
    def vibration2tensor(vib_list, channel):
        # vib_list: [sig] or [sig1,sig3,sig5]
        # output [3,1,len] or [1,len]
        vib_tensor = []
        channel = channel
        for vib in vib_list:
            m1, m2 = vib.min(), vib.max()
            vib_n = (vib - m1) / (m2 - m1)  # change from vib-vib.mean()
            vib_nt = torch.from_numpy(vib_n)
            vib_tensor.append(vib_nt)
        vibration_tensor = torch.stack(vib_tensor, dim=0)
        if channel == 1:
            vibration_tensor = torch.squeeze(vibration_tensor, dim=0)
            vibration_tensor = torch.unsqueeze(vibration_tensor, dim=0)
        elif channel == 3:
            vibration_tensor = torch.unsqueeze(vibration_tensor, dim=1)
        return vibration_tensor

    @staticmethod
    def time_shift(vib_t, shift_limit=1, sr=500):
        vib_len = vib_t.shape[-1]
        shift_amt = int(random.random() * shift_limit * vib_len)
        return vib_t.roll(shift_amt), shift_amt

    @staticmethod
    def padding2(vib_t, channel, max_point=484):
        len0 = vib_t.shape[-1]
        len1_ = vib_t.shape[:-1]
        len1 = [_ for _ in len1_]
        len2 = len1.copy()
        if len0 == max_point:
            return vib_t
        elif len0 < max_point:
            pad_end_len = max_point - len0
            if channel == 3:
                padded_vib = F.pad(vib_t, (0, pad_end_len), 'replicate')
            elif channel == 1:
                len2.append(pad_end_len)
                pad_end = torch.ones(len2)
                pad_end = torch.mul(pad_end, vib_t[..., -1])
                padded_vib = torch.cat((vib_t, pad_end), -1)
        elif len0 > max_point:
            padded_vib = vib_t[..., 0:max_point]
        return padded_vib

    @staticmethod
    def spectrogram(vib_t, n_fft=20, win_len=20, normalized=True):
        # return tensor shape [channel,freq_bin, time_frame] [1,101,101]
        spec_ = transforms.Spectrogram(n_fft=n_fft, win_length=win_len, window_fn=torch.hann_window, normalized=True)(
            vib_t)
        spec = torch.squeeze(spec_, 1)
        return spec

