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
from vibdataset import VibrationDS, VibrationDS2
from vibmodel import ModelTest4_plus
from vibtrain import training

if __name__ == '__main__':
    # load data
    metadata_path = 'D:/graduate_project/data2/metadata/vehicleclassification7.csv'
    data_path = 'D:/graduate_project/data2/data'
    # my_vib_ds = VibrationDS(metadata_path, data_path, 1, True, False)
    my_vib_ds = VibrationDS2(metadata_path, data_path, 1, True, False)
    num_item = len(my_vib_ds)
    num_train = round(num_item * 0.8)
    # num_train = 480
    # num_val = 128
    num_val = num_item - num_train
    train_ds, val_ds = random_split(my_vib_ds, [num_train, num_val])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=True)

    # split data into 5 part
    num1 = num2 = num3 = num4 = round(0.2 * num_item)
    num5 = num_item - num1 - num2 - num3 - num4
    set1, set2, set3, set4, set5 = random_split(my_vib_ds, [num1, num2, num3, num4, num5],
                                                generator=torch.Generator().manual_seed(42))
    train_list = [set2, set3, set4, set5]
    train_ds = torch.utils.data.ConcatDataset(train_list)
    val_ds = set1
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=True)

    # create model
    my_vib_classifier = ModelTest4_plus()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_vib_classifier = my_vib_classifier.to(device)
    next(my_vib_classifier.parameters()).device

    training(my_vib_classifier, train_dl, val_dl, 200)
