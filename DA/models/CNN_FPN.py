import torch.nn as nn
import torch
import pdb

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN_FPN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(CNN_FPN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

        self.cnn = cnn

        # element for fpn
        self.cnn_fcn = nn.Conv2d(128,128,3,1,1)
        self.glu = GLU(128)
        self.pool_fcn = nn.AvgPool2d([2,1])
        self.deconv1 = nn.Upsample((78,1), mode='bilinear', align_corners=True)#nn.ConvTranspose2d(128,128,kernel_size=[2,1],stride=[2,1])
        self.deconv2 = nn.Upsample((157,1), mode='bilinear', align_corners=True)#nn.ConvTranspose2d(128,128,kernel_size=[2,1],stride=[2,1],output_padding=[1,0])
        self.bn_fcn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.conv1x1 = nn.Conv2d(256,128,1)
        self.dropout = nn.Dropout(0.5) 


    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        # further pooling for fcn
        x_up = self.cnn_fcn(x) 
        x_up = self.bn_fcn(x_up)
        x_up = self.glu(x_up)        
        x_up = self.dropout(x_up)
        x_up = self.pool_fcn(x_up) 
        x_2 = x_up # 128 x 78 x 1
        x_up = self.cnn_fcn(x_up)
        x_up = self.bn_fcn(x_up)
        x_up = self.glu(x_up)
        x_up = self.dropout(x_up)
        x_up = self.pool_fcn(x_up) 
        x_4 = x_up # 128 x 39 x 1

        return x, x_2, x_4


