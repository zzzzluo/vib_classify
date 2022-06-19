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


def training(model, train_dl, val_dl, num_epochs, lr=0.001, weight_decay=0.0001):
    val_loss_min = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    # repeat for each
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    metrics_dir = 'D:/graduate_project/data2/metrics'
    metrics_file = '{}/loss_metrics_{}.txt'.format(metrics_dir, current_time)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        validation_loss = 0.0
        correct_prediction_val = 0
        total_prediction_val = 0

        # repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            model.train()
            # get the input features and target labels and put them on the gpu
            inputs, labels = data[0].to(device), data[1].to(device)

            # normalize the inputs
            inputs = inputs.float()  # otherwise mean() error

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # test
            # inputs = inputs.unsqueeze(3)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward backward optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # keep stats for loss and accuracy
            running_loss += loss.item()

            # get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}. Accuracy:{acc:2f}')

        # model eval
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                model.eval()
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.float()
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s
                output = model(inputs)
                loss = criterion(output, labels)
                validation_loss += loss.item()

                # get the predicted class with the highest score
                _, prediction_val = torch.max(output, 1)
                correct_prediction_val += (prediction_val == labels).sum().item()
                total_prediction_val += prediction_val.shape[0]

        num_val = len(val_dl)
        avg_validation_loss = validation_loss / num_val
        acc_val = correct_prediction_val / total_prediction_val
        print(f'validationloss:{avg_validation_loss}, validation accuracy: {acc_val:2f}')

        if avg_validation_loss < val_loss_min:
            best_models_dir = 'D:/graduate_project/data2/best_model_weights'
            best_model_weights_filename = '{}/best_model_weights_{}_loss_{}.pth'.format(best_models_dir, current_time,
                                                                                        avg_validation_loss)
            state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state_dict, best_model_weights_filename)
            print("Best Model weights saved in file:", best_model_weights_filename)
            val_loss_min = avg_validation_loss

        elif epoch % 200 == 0:
            checkpoint_dir = 'D:/graduate_project/data2/checkpoint'
            weights_filename = '{}/checkpoint_weights_{}_epoch_{}.pth'.format(checkpoint_dir, current_time, epoch)

            state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state_dict, weights_filename)
            print("checkpoint weights saved in file:", weights_filename)

        with open(metrics_file, 'a') as f_metrics_file:
            f_metrics_file.write(
                '%d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n' % (epoch + 1, avg_loss, avg_validation_loss, acc, acc_val))

    print('Finished Training')
    # save the model
    models_dir = 'D:/graduate_project/data2/model_weights'
    model_weights_filename = '{}/model_weights_{}_{}.pth'.format(models_dir, current_time, num_epochs)
    state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state_dict, model_weights_filename)
    print("Model weights saved in file:", model_weights_filename)
