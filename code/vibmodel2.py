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
# from vibdataset import VibrationDS, VibrationDS2
from vibmodel import ModelTest4_plus
from vibtrain import training
from vibdataset2 import VibrationDS, VibrationDS2
from vibpredict import testing


# 2d residual net
class ResBlock2d(torch.nn.Module):
    # input two conv module
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)):
        # shape: n,c,h,w

        super().__init__()
        conv_layers = []
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 1st main seperable conv
        self.depth_conv0 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, groups=self.in_channel)
        self.point_conv0 = nn.Conv2d(self.in_channel, self.mid_channel,
                                     kernel_size=(1, 1))  # stride and padding should remain default which is 1 and 0
        self.relu0 = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(self.mid_channel)
        init.kaiming_normal_(self.depth_conv0.weight, a=0.1)
        self.depth_conv0.bias.data.zero_()
        init.kaiming_normal_(self.point_conv0.weight, a=0.1)
        self.point_conv0.bias.data.zero_()
        self.conv0 = nn.Sequential(self.depth_conv0, self.point_conv0, self.relu0, self.bn0)
        # 2nd main seperable conv
        self.depth_conv1 = nn.Conv2d(self.mid_channel, self.mid_channel, kernel_size=self.kernel_size,
                                     stride=self.stride, padding=self.padding, groups=self.mid_channel)
        self.point_conv1 = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        init.kaiming_normal_(self.depth_conv1.weight, a=0.1)
        self.depth_conv1.bias.data.zero_()
        init.kaiming_normal_(self.point_conv1.weight, a=0.1)
        self.point_conv1.bias.data.zero_()
        self.conv1 = nn.Sequential(self.depth_conv1, self.point_conv1, self.bn1)
        # branch seperable conv
        self.depth_conv2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, groups=self.in_channel)
        self.point_conv2 = nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=(1, 1))
        # self.mp2 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        init.kaiming_normal_(self.depth_conv2.weight, a=0.1)
        self.depth_conv2.bias.data.zero_()
        init.kaiming_normal_(self.point_conv2.weight, a=0.1)
        self.point_conv2.bias.data.zero_()
        self.bn2 = nn.BatchNorm2d(self.mid_channel)

        self.depth_conv3 = nn.Conv2d(self.mid_channel, self.mid_channel, kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding, groups=self.mid_channel)
        self.point_conv3 = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        init.kaiming_normal_(self.depth_conv3.weight, a=0.1)
        self.depth_conv3.bias.data.zero_()
        init.kaiming_normal_(self.point_conv3.weight, a=0.1)
        self.point_conv3.bias.data.zero_()
        self.conv2 = nn.Sequential(self.depth_conv2, self.point_conv2, self.bn2, self.depth_conv3, self.point_conv3,
                                   self.bn3)

    @staticmethod
    def ds_vib(n1, c1, h1, w1, x):
        n0, c0, h0, w0 = x.shape
        mp = nn.AdaptiveMaxPool2d((h1, w1))
        x = mp(x)
        return x

    @staticmethod
    def shape_calculator(module1, module2, h_in, w_in):
        h_out0 = int(
            (h_in + 2 * module1.padding[0] - module1.dilation[0] * (module1.kernel_size[0] - 1) - 1) / module1.stride[
                0] + 1)
        w_out0 = int(
            (w_in + 2 * module1.padding[1] - module1.dilation[1] * (module1.kernel_size[1] - 1) - 1) / module1.stride[
                1] + 1)
        h_out = int(
            (h_out0 + 2 * module2.padding[0] - module2.dilation[0] * (module2.kernel_size[0] - 1) - 1) / module2.stride[
                0] + 1)
        w_out = int(
            (w_out0 + 2 * module2.padding[1] - module2.dilation[1] * (module2.kernel_size[1] - 1) - 1) / module2.stride[
                1] + 1)
        return h_out, w_out

    def forward(self, inputs):
        inputs1 = self.conv0(inputs)
        inputs1 = self.conv1(inputs1)
        # n1,c1,h1,w1 = inputs.shape
        inputs2 = self.conv2(inputs)
        # inputs_ = self.ds_vib(n1,c1,h1,w1,inputs_)
        out = inputs1 + inputs2
        out = self.relu1(out)
        return out


class ModelTest2_4_Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (3, 3))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(1)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.conv = nn.Sequential(self.conv1, self.relu1, self.bn1)

        self.rb1 = ResBlock2d(1, 8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.mp0 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.rb2 = ResBlock2d(16, 32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.mp1 = nn.AvgPool2d(3, stride=2)
        self.rb3 = ResBlock2d(64, 128, 256, kernel_size=(3, 3), stride=(1, 2), padding=(0, 0))
        self.mp2 = nn.AvgPool2d(3, stride=2)
        self.rb4 = ResBlock2d(256, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.mp3 = nn.AvgPool2d(3, stride=2)
        self.rb5 = ResBlock2d(256, 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.mp4 = nn.AvgPool2d(3, stride=1)
        self.rb = nn.Sequential(self.rb1, self.mp0, self.rb2, self.mp1, self.rb3, self.mp2, self.rb4, self.mp3,
                                self.rb5, self.mp4)
        self.rbe = nn.Sequential(self.rb1, self.rb2, self.rb3, self.rb4, self.rb5)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(in_features=512, out_features=5)
        self.dp1 = nn.Dropout2d(p=0.1)
        self.dp2 = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv(x)
        x = self.rbe(x)
        x = self.ap(x)
        # x = self.bn(x)
        # x = self.dp1(x)
        x = x.view(x.shape[0], -1)
        x = self.dp2(x)
        x = self.lin(x)

        return x


if __name__ == '__main__':
    Train = True
    Test = False

    if Train:
        metadata_path = 'D:/graduate_project/data2/metadata/vehicleclassification7.csv'
        data_path = 'D:/graduate_project/data2/data'
        # my_vib_ds = VibrationDS(metadata_path, data_path, 1, True, False)
        my_vib_ds = VibrationDS2(metadata_path, data_path, 1, True, True)
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
        train_list = [set1, set3, set4, set5]
        train_ds = torch.utils.data.ConcatDataset(train_list)
        val_ds = set2
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=True)

        # create model
        my_vib_classifier = ModelTest2_4_Res()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        my_vib_classifier = my_vib_classifier.to(device)
        next(my_vib_classifier.parameters()).device

        training(my_vib_classifier, train_dl, val_dl, 200, lr=0.0001, weight_decay=0.001)

    if Test:
        my_vib_classifier = ModelTest2_4_Res()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        my_vib_classifier = my_vib_classifier.to(device)
        next(my_vib_classifier.parameters()).device

        test_metadata_file = 'D:/graduate_project/data2/metadata/vehicleclassification_test7.csv'
        data_path = 'D:/graduate_project/data2/data'
        # my_vib_ds = VibrationDS(metadata_path, data_path, 1, False, False)
        test_ds = VibrationDS(test_metadata_file, data_path, 1, True, True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
        weight_file = 'D:/graduate_project/data2/test_model_weights/best_model_weights_2022_06_02_00_57_03_loss_0.10634292396051544.pth'
        testing(my_vib_classifier, test_dl, weight_file)
