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


def testing(my_vib_classifier, test_dl, model_weights_filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_model_file = model_weights_filename
    state_dict = torch.load(init_model_file, map_location=device)
    my_vib_classifier.load_state_dict(state_dict['model_state_dict'])  # vib_model -> my_vib_classifier

    correct_prediction = 0
    total_prediction = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # test
    with torch.no_grad():
        for inputs, labels in test_dl:
            my_vib_classifier.eval()
            input = inputs.to(device)
            labels = labels.to(device)
            input = input.float()
            input_m, input_s = input.mean(), input.std()
            input = (input - input_m) / input_s
            outputs = my_vib_classifier(input)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            print('ground truth', labels)
            print('prediction', prediction)
        acc = correct_prediction / total_prediction
        print(f'Acc:{acc:2f},Total items:{total_prediction}')
        print("average_loss:", test_loss / len(test_dl))


if __name__ == '__main__':
    my_vib_classifier = ModelTest4_plus()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_vib_classifier = my_vib_classifier.to(device)
    next(my_vib_classifier.parameters()).device

    test_metadata_file = 'D:/graduate_project/data2/metadata/vehicleclassification_test7.csv'
    data_path = 'D:/graduate_project/data2/data'
    # my_vib_ds = VibrationDS(metadata_path, data_path, 1, False, False)
    test_ds = VibrationDS(test_metadata_file, data_path, 1, True, False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
    weight_file = 'D:/graduate_project/data2/test_model_weights/best_model_weights_2022_05_09_03_29_39_loss_0.01147463486995548 (1).pth'
    testing(my_vib_classifier, test_dl, weight_file)
