################################################
########        DESIGN   NETWORK        ########
################################################

import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,kernel_size=(1,3),stride=1, padding=0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,kernel_size=(1,3), stride=1, padding=0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d((1,2), ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs



"""
class  NetModel_r(nn.Module):
    def __init__(self, in_channels ,is_deconv, is_batchnorm, data_dim, label_dim):
        super(NetModel_r, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.flatten       = Flatten()
        self.dropout       = nn.Dropout(p=0.3)
        filters = [8, 16, 32, 64, 64]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.center  = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.mlp1     = nn.Linear(306*filters[3],filters[4])
        self.mlp2    = nn.Linear(filters[4], 1)
        
    def forward(self, inputs,label_dsp_dim):
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        center = self.center(down3)
        x      = self.flatten(center)
        mlp    = self.mlp1(x).reshape([x.size(0),-1])
        mlp    = self.mlp2(mlp).reshape([x.size(0),-1])
        return mlp.contiguous()
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()
"""

class  NetModel_r(nn.Module):
    def __init__(self, in_channels ,is_deconv, is_batchnorm, data_dim, bubble_num):
        super(NetModel_r, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.flatten       = Flatten()
        self.dropout       = nn.Dropout(p=0.3)

        self.mlp    = nn.Linear(data_dim[1], bubble_num)
        
    def forward(self, inputs):
        mlp    = self.mlp(inputs).reshape([inputs.size(0),-1])
        return mlp.contiguous()
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

class  NetModel_v(nn.Module):
    def __init__(self, in_channels ,is_deconv, is_batchnorm, data_dim, label_dim):
        super(NetModel_v, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.flatten       = Flatten()
        self.dropout       = nn.Dropout(p=0.3)
        filters = [8, 16, 32, 64]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.center  = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.mlp1     = nn.Linear(306*filters[3], label_dim[1])
        
    def forward(self, inputs,label_dsp_dim):
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        center = self.center(down3)
        center = center.reshape(center.size(0),-1)
        mlp    = self.mlp1(center).reshape([center.size(0),-1])
        
        return mlp.contiguous()
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()
