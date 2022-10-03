import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import sys

from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim, device):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1. / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, ndomains):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)  # disable for digits
    self.ad_layer3 = nn.Linear(hidden_size, ndomains)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()  # disable for digits
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)  # disable for digits
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    self.ndomains = ndomains

  def output_num(self):
    return self.ndomains

  def get_parameters(self):
    return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

  def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
      return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

  def grl_hook(self, coeff):
      def fun1(grad):
          return -coeff * grad.clone()

      return fun1

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = self.calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    if self.training:
        x.register_hook(self.grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)  # disable for digits
    x = self.relu2(x)  # disable for digits
    x = self.dropout2(x)  # disable for digits
    y = self.ad_layer3(x)
    return y




class ViT(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ViT, self).__init__()

        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 100
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        config_vit.pretrained_path = './pretrainedViT/R50+ViT-B_16.npz'
        self.feature_layers = ViT_seg(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
        self.feature_layers.load_from(weights=np.load(config_vit.pretrained_path))
        
        vitFeatureNumber = 2048



        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(vitFeatureNumber, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(vitFeatureNumber, class_num)
                self.fc.apply(init_weights)
                self.__in_features = vitFeatureNumber
        else:
            sys.exit('This condition has not been defined for the ViT model')

    def forward(self, x):
        _,x = self.feature_layers(x)
        #x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
            else:
                parameter_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                    {'params': self.fc.parameters(), 'lr_mult': 10, 'decay_mult': 2}
                ]
        else:
            parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list
