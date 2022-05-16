import csv
import gzip
import itertools
import json, array
import os
import random, zipfile
import math, scipy.signal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import utils, am_decoder
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time
import json
import matplotlib.pyplot as plt
import preprocess as pp
from utils import get_dataloaders

from memspec import *
import memspec.GenerateTimeSeries as GenerateTimeSeries

import warnings

warnings.filterwarnings("ignore")


class MdDataset(Dataset):
    def __init__(self, meta_data, type='train', transform=None):
        self.meta_data = meta_data
        self.transform = transform
        print(f'{type.capitalize()}_dataset load samples: {len(meta_data)}')

    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        meta_data = self.meta_data[index]
        dexs = meta_data['dex']
        markov_img = Image.open(dexs['markov']['classes'])
        loc_img = Image.open(dexs['loc']['classes']).convert('RGB')
        byte_img = Image.open(dexs['byte']['classes']).convert('RGB')
        markov_img = self.transform(markov_img)
        loc_img = self.transform(loc_img)
        byte_img = self.transform(byte_img)
        return byte_img, loc_img, markov_img, meta_data['is_mal']

    def __len__(self):
        return len(self.meta_data)


def md_collate_fn(data):
    byte_img, loc_img, markov_img, is_mal = zip(*data)
    is_mal = torch.LongTensor(is_mal)
    byte_img = torch.stack(byte_img)
    loc_img = torch.stack(loc_img)
    markov_img = torch.stack(markov_img)
    return (byte_img, loc_img, markov_img, is_mal)


class EnDataset(Dataset):
    def __init__(self, meta_data, type='train', transform=None, fs=1.0):
        self.meta_data = meta_data
        self.transform = transform
        self.fs = fs
        self.type = 'header'
        print(f'{type.capitalize()}_dataset load samples: {len(meta_data)}')

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        dexs = meta_data['dex']
        entropy_src = dexs['entropy']['classes']
        mat_src = dexs[self.type]['classes']
        entropy = np.load(entropy_src)
        mat = np.load(mat_src)
        f, Pxx = scipy.signal.welch(entropy, 5)
        Pxx = torch.tensor(Pxx).float()
        mat = torch.tensor(mat).float()
        return Pxx, mat, meta_data['is_mal']

    def __len__(self):
        return len(self.meta_data)


def en_collate_fn(data):
    entropy, header, is_mal = zip(*data)
    is_mal = torch.LongTensor(is_mal)
    entropy = rnn_utils.pad_sequence(entropy, batch_first=True)
    header = torch.stack(header)
    return (entropy, header, is_mal)


class FMSDataset(Dataset):
    def __init__(self, meta_data, type='train', dict_len=0, fs=1.0):
        self.meta_data = meta_data
        self.field_name = 'header'
        self.dict_len = dict_len
        print(f'{type.capitalize()}_dataset load samples: {len(meta_data)}')

    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        meta_data = self.meta_data[index]
        dexs = meta_data['dex']
        #header
        header = np.load(dexs['header']['classes'])

        #Entropy
        entropy = np.load(dexs['entropy']['classes'])
        dt = 5
        M = MESA()
        P, ak, opt = M.solve(entropy,
                             method="standard",
                             optimisation_method="Fixed")
        f_PSD = np.linspace(0, 0.1, 129)
        Pxx = M.spectrum(dt, f_PSD)[1:]
        if len(Pxx) < 128:
            Pxx = np.pad(Pxx, (0, 128 - len(Pxx)), 'constant')
        else:
            Pxx = Pxx[:128]
        Pxx = np.pad(Pxx, (0, 129 - len(Pxx)), 'constant')[:-1]
        header = np.pad(header, (0, 128 - len(header)), 'constant')

        #Permission&Intent
        ip = None
        if not meta_data['intent_permission']:
            ip = np.zeros(self.dict_len)
        else:
            ip = np.load(meta_data['intent_permission'])

        header = torch.tensor(header).float()
        Pxx = torch.tensor(Pxx).float()
        ip = torch.tensor(ip).float()
        return header, Pxx, ip, meta_data['is_mal']

    def __len__(self):
        return len(self.meta_data)


def fms_collate_fn(data):
    header, pxx, ip, is_mal = zip(*data)
    is_mal = torch.LongTensor(is_mal)
    header = torch.stack(header)
    ip = torch.stack(ip)
    #pxx = rnn_utils.pad_sequence(pxx, batch_first=True)
    pxx = torch.stack(pxx)
    return (header, pxx, ip, is_mal)


class PreDataset(Dataset):
    def __init__(self, data=None, dictionary=None, fs=1.0):
        self.fs = fs
        self.data = data
        self.dictionary = dictionary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, is_mal = self.data[index]
        am_file, classes_file, apk_size = utils.extract_apk(file_path)
        classes_data = np.array(array.array('B', classes_file))
        header = classes_data[:0x70] / 256
        entropy = utils.get_entropy(classes_data).astype(np.float64)
        dt = 5
        M = MESA()
        P, ak, opt = M.solve(entropy,
                             method="standard",
                             optimisation_method="Fixed")
        f_PSD = np.linspace(0, 0.1, 129)
        Pxx = M.spectrum(dt, f_PSD)[1:]
        if len(Pxx) < 128:
            Pxx = np.pad(Pxx, (0, 128 - len(Pxx)), 'constant')
        else:
            Pxx = Pxx[:128]
        header = np.pad(header, (0, 128 - len(header)), 'constant')
        ip = utils.get_ip_feature(am_file, self.dictionary)
        #print('ip', (time.clock() - start) * 1000)
        header = torch.tensor(header).float()
        Pxx = torch.tensor(Pxx).float()
        Pxx = torch.nn.functional.normalize(Pxx, dim=0)
        ip = torch.tensor(ip).float()
        return header, Pxx, ip, is_mal


def pre_collate_fn(data):
    header, pxx, ip, is_mal = zip(*data)
    is_mal = torch.LongTensor(is_mal)
    header = torch.stack(header)
    ip = torch.stack(ip)
    #pxx = rnn_utils.pad_sequence(pxx, batch_first=True)
    pxx = torch.stack(pxx)
    return (header, pxx, ip, is_mal)
