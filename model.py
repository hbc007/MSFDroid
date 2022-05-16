import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from dynamic_conv import Dynamic_conv1d

from resnet import resnet50


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  #CW
        # average = x    #CS
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class ASU_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_dConv1d = Dynamic_conv1d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=stride,
                                               padding=1,
                                               bias=False)
        self.shortcut_dConv1d = Dynamic_conv1d(in_channels,
                                               out_channels *
                                               ASU_Block.expansion,
                                               kernel_size=1,
                                               stride=stride,
                                               bias=False)
        self.conv2 = Dynamic_conv1d(out_channels,
                                    out_channels * ASU_Block.expansion,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.residual_function = nn.Sequential(
            self.residual_dConv1d, nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), self.conv2,
            nn.BatchNorm1d(out_channels * ASU_Block.expansion), self.shrinkage)
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ASU_Block.expansion * out_channels:
            self.shortcut = nn.Sequential(
                self.shortcut_dConv1d,
                nn.BatchNorm1d(out_channels * ASU_Block.expansion))

    def update_temperature(self):
        self.residual_dConv1d.update_temperature()
        self.shortcut_dConv1d.update_temperature()
        self.conv2.update_temperature()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) +
                                     self.shortcut(x))

    # a = self.residual_function(x),
    # b = self.shortcut(x),
    # c = a+b
    # return c


class Model(nn.Module):
    def __init__(self,
                 models='hpic',
                 ada_voting=False,
                 vote_threshold=1.0,
                 device=torch.device("cpu")):
        super(Model, self).__init__()
        self.in_channels = 64
        self.mlp_embed_size = 128
        self.asu_layers = []
        self.models = models
        self.criterion = nn.BCELoss()
        self.device = device
        self.ada_voting = ada_voting

        # Header Model
        self.header_mlp = None
        if 'h' in self.models:
            self.header_mlp = self.init_mlp(128, 2)

        # Pxx Model
        self.pxx_mlp = None
        if 'p' in self.models:
            self.pxx_mlp = self.init_mlp(128, 2)

        # Intent & Permission
        self.dConv1 = None
        self.conv1 = None
        self.conv2 = None
        if 'i' in self.models:
            self.dConv1 = Dynamic_conv1d(1,
                                         self.in_channels,
                                         kernel_size=3,
                                         padding=1,
                                         stride=2,
                                         bias=False)
            self.conv1 = nn.Sequential(self.dConv1,
                                       nn.BatchNorm1d(self.in_channels),
                                       nn.ReLU(inplace=True))
            self.conv2 = self.add_asu_layer(128, 2, 2)
            self.ip_mlp = self.init_mlp(128, 2)

        # Combine Model
        self.vote_threshold = vote_threshold
        self.cb_dConv1 = None
        self.cb_conv1 = None
        self.cb_conv2 = None
        self.cb_dropout = None
        self.cb_mlp = None
        if 'c' in self.models:
            self.cb_dConv1 = Dynamic_conv1d(1,
                                            self.in_channels,
                                            kernel_size=3,
                                            padding=1,
                                            stride=2,
                                            bias=False)
            self.cb_conv1 = nn.Sequential(self.cb_dConv1, nn.BatchNorm1d(128),
                                          nn.ReLU(inplace=True))
            self.cb_conv2 = self.add_asu_layer(128, 2, 2)
            self.cb_dropout = nn.Dropout(0.1)
            self.cb_mlp = self.init_mlp(128, 2)

        # Attention Model
        self.attention = nn.Sequential(
            nn.Linear(128 * len(self.models), len(self.models)),
            nn.BatchNorm1d(len(self.models)),
            nn.ReLU(inplace=True),
            nn.Linear(len(self.models), len(self.models)),
            nn.Softmax(),
        )
        # Other
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.max_pool = nn.AdaptiveMaxPool1d((1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.zero = torch.zeros(1, requires_grad=False).to(self.device)
        self.voting_weight = nn.Parameter(torch.ones(len(models)),
                                          requires_grad=ada_voting)

    def get_vote_weight(self):
        return self.softmax(self.voting_weight)

    def init_mlp(self, embed_size, layers, output_size=1):
        predictor = []
        for _ in range(layers + 1):
            linear = nn.Linear(embed_size, embed_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            predictor.append(linear)
            predictor.append(nn.ReLU())
        linear = nn.Linear(embed_size, output_size)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0)
        predictor.append(linear)
        return nn.Sequential(*predictor)

    def add_asu_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            self.asu_layers.append(
                ASU_Block(self.in_channels, out_channels, stride))
            layers.append(self.asu_layers[-1])
            self.in_channels = out_channels * ASU_Block.expansion

        return nn.Sequential(*layers)

    def update_temperature(self):
        if 'i' in self.models:
            self.dConv1.update_temperature()
        if 'c' in self.models:
            self.cb_dConv1.update_temperature()
        for layer in self.asu_layers:
            layer.update_temperature()

    def forward(self, header, pxx, ip, is_mal):
        header = header.to(self.device)
        ip = ip.to(self.device)
        pxx = pxx.to(self.device)
        target = is_mal.to(self.device)

        predict_list = []
        loss_list = []
        feature_list = []
        if 'h' in self.models:
            feature_list.append(header)
            header_predict = self.sigmoid(self.header_mlp(header))
            predict_list.append(header_predict)
            header_loss = self.criterion(header_predict.squeeze(dim=1), target)
            loss_list.append(header_loss)
        else:
            header_predict = self.zero
            header_loss = self.zero

        if 'p' in self.models:
            feature_list.append(pxx)
            pxx_predict = self.sigmoid(self.pxx_mlp(pxx))
            predict_list.append(pxx_predict)
            pxx_loss = self.criterion(pxx_predict.squeeze(dim=1), target)
            loss_list.append(pxx_loss)
        else:
            pxx_predict = self.zero
            pxx_loss = self.zero

        if 'i' in self.models:
            ip = ip.unsqueeze(-2)
            ip = self.conv1(ip)
            ip = self.conv2(ip)
            ip = self.avg_pool(ip).squeeze(-1)
            feature_list.append(ip)
            ip_predict = self.sigmoid(self.ip_mlp(ip))
            predict_list.append(ip_predict)
            ip_loss = self.criterion(ip_predict.squeeze(dim=1), target)
            loss_list.append(ip_loss)
        else:
            ip_predict = self.zero
            ip_loss = self.zero

        if 'c' in self.models:
            cb = torch.cat((header, pxx), dim=-1).unsqueeze(-2)
            cb = self.cb_conv1(cb)
            cb = self.cb_dropout(cb)
            cb = self.cb_conv2(cb)
            cb = self.max_pool(cb).squeeze(-1)
            feature_list.append(cb)
            cb_predict = self.sigmoid(self.cb_mlp(cb))
            predict_list.append(cb_predict)
            cb_loss = self.criterion(cb_predict.squeeze(dim=1), target)
            loss_list.append(cb_loss)
        else:
            cb_predict = self.zero
            cb_loss = self.zero

        sv_predict = torch.cat(predict_list,
                               dim=1).mul(self.softmax(self.voting_weight))
        sv_predict = sv_predict.sum(1)
        if self.ada_voting:
            sv_loss = self.criterion(sv_predict, target) + torch.abs(
                self.softmax(-torch.stack(loss_list)) -
                self.softmax(self.voting_weight)).mean() * self.vote_threshold
        else:
            sv_loss = self.zero

        return (header_predict,
                header_loss), (pxx_predict,
                               pxx_loss), (ip_predict,
                                           ip_loss), (cb_predict,
                                                      cb_loss), (sv_predict,
                                                                 sv_loss)
