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


class ResBlock4(torch.nn.Module):
    # input two conv module
    def __init__(self, module1, module2, channel1, channel2):
        # shape: n,c,h,w

        super().__init__()
        conv_layers = []
        conv_layers_ = []
        self.channel1 = channel1
        self.channel2 = channel2
        self.module1 = module1
        self.module1_ = module1
        self.relu1 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(self.channel1)
        self.bn1_ = nn.BatchNorm1d(self.channel1)
        init.kaiming_normal_(self.module1.weight, a=0.1)
        init.kaiming_normal_(self.module1_.weight, a=0.1)
        self.module1.bias.data.zero_()
        self.module1_.bias.data.zero_()
        conv_layers += [self.module1, self.relu1, self.bn1]
        # conv_layers += [self.module1, self.bn1, self.relu1]
        conv_layers_ += [self.module1_, self.bn1_]

        self.module2 = module2
        self.module2_ = module2
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.channel2)
        self.bn2_ = nn.BatchNorm1d(self.channel2)
        init.kaiming_normal_(self.module2.weight, a=0.1)
        self.module2.bias.data.zero_()
        init.kaiming_normal_(self.module2_.weight, a=0.1)
        self.module2_.bias.data.zero_()
        conv_layers += [self.module2, self.bn2]
        conv_layers_ += [self.module2_, self.bn2_]

        self.conv = nn.Sequential(*conv_layers)
        self.conv_ = nn.Sequential(*conv_layers_)

    @staticmethod
    def shape_calculator(module1, module2, h_in):
        h_out0 = int(
            (h_in + 2 * module1.padding[0] - module1.dilation[0] * (module1.kernel_size[0] - 1) - 1) / module1.stride[
                0] + 1)

        h_out = int(
            (h_out0 + 2 * module2.padding[0] - module2.dilation[0] * (module2.kernel_size[0] - 1) - 1) / module2.stride[
                0] + 1)
        return h_out

    def forward(self, inputs):
        inputs1 = self.conv(inputs)
        inputs2 = self.conv_(inputs)
        out = inputs1 + inputs2
        outs = self.relu2(out)
        return outs


class ModelTest4_plus(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 1st
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]
        # conv_layers += [self.conv1, self.bn1, self.relu1]

        self.conv = nn.Sequential(*conv_layers)
        self.module1 = nn.Conv1d(8, 16, 3)
        self.module2 = nn.Conv1d(16, 32, 3)
        self.module3 = nn.Conv1d(32, 64, 3)
        self.module4 = nn.Conv1d(64, 128, 3, stride=2)
        self.module5 = nn.Conv1d(128, 256, 3, stride=2)
        self.module6 = nn.Conv1d(256, 256, 3, stride=2)
        self.module7 = nn.Conv1d(256, 256, 3, stride=2)
        self.module8 = nn.Conv1d(256, 256, 3, stride=2)
        self.module9 = nn.Conv1d(256, 512, 3, stride=2)
        self.module10 = nn.Conv1d(512, 512, 3, stride=2)
        self.rb1 = ResBlock4(self.module1, self.module2, 16, 32)
        self.rb2 = ResBlock4(self.module3, self.module4, 64, 128)
        self.rb3 = ResBlock4(self.module5, self.module6, 256, 256)
        self.rb4 = ResBlock4(self.module7, self.module8, 256, 256)
        self.rb5 = ResBlock4(self.module9, self.module10, 512, 512)

        self.rb = nn.Sequential(self.rb1, self.rb2, self.rb3, self.rb4, self.rb5)
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Linear(in_features=512, out_features=5)

    def forward(self, x):
        x = self.conv(x)
        x = self.rb(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x


if __name__ == '__main__':
    print('test')
