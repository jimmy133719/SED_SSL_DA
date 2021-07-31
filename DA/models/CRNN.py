import warnings

import torch.nn as nn
import torch
from torch.autograd import Function

from models.RNN import BidirectionalGRU
from models.CNN import CNN
from models.CNN_FPN import CNN_FPN
# from RNN import BidirectionalGRU
# from CNN import CNN
# from CNN_FPN import CNN_FPN

import pdb

class Clip_Discriminator(nn.Module):
    def __init__(self, input_dim, dropout=0):
        super(Clip_Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 128, kernel_size=3, stride=2)
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=3, stride=2)
        self.conv_3 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.conv_4 = nn.Conv2d(32, 16, kernel_size=3, stride=2)
        self.conv_5 = nn.Conv2d(16, 8, kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((2,1))
        self.dense_d = nn.Linear(16, 2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)
        
        self.bn_1 = nn.BatchNorm2d(128)
        self.bn_2 = nn.BatchNorm2d(64)
        self.bn_3 = nn.BatchNorm2d(32)
        self.bn_4 = nn.BatchNorm2d(16)
        self.bn_5 = nn.BatchNorm2d(8) 
 

    def forward(self, x):
        x = torch.unsqueeze(x.permute(0, 2, 1), 1)
        x = self.leaky_relu(self.bn_1(self.conv_1(x)))
        x = self.leaky_relu(self.bn_2(self.conv_2(x)))
        x = self.leaky_relu(self.bn_3(self.conv_3(x)))
        x = self.leaky_relu(self.bn_4(self.conv_4(x)))
        x = self.leaky_relu(self.bn_5(self.conv_5(x)))
        x = self.avgpool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.dense_d(x)
        domain_out = self.softmax(x)

        return domain_out


class Frame_Discriminator(nn.Module):
    def __init__(self, input_dim, dropout=0):
        super(Frame_Discriminator, self).__init__()
        self.dense_d_1 = nn.Linear(input_dim, 128)
        self.dense_d_2 = nn.Linear(128, 32)
        self.dense_d_3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # self.avgpool = nn.AvgPool2d((2,4))
        # self.bn_1 = nn.BatchNorm1d(128)
        # self.bn_2 = nn.BatchNorm1d(32) 

    def forward(self, x):
        x = self.leaky_relu(self.dense_d_1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.dense_d_2(x))
        x = self.dropout(x)
        x = self.dense_d_3(x)
        x = self.softmax(x)
        domain_out = x

        return domain_out

class CRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0,
                 cnn_integration=False, learned_post=False, **kwargs):
        super(CRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.rnn_type = rnn_type
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if self.rnn_type == 'BGRU':
            nb_in = self.cnn.nb_filters[-1] 
            if self.cnn_integration:
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)

            
        elif self.rnn_type =='TCN':
            # Number of [n_RNN_cell] needs to be defined
            self.rnn = TemporalConvNet(self.cnn.nb_filters[-1], [n_RNN_cell] * 2, 3, dropout=0.25)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)
        
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)        
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        # rnn features
        if self.rnn_type == 'BGRU':
            x = self.rnn(x)
        elif self.rnn_type == 'TCN':
            x = self.rnn(x.transpose(1, 2)).transpose(1, 2)

        x = self.dropout(x)
        d_input = x 

        return x, d_input


class CRNN_fpn(nn.Module):
    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0,
                 cnn_integration=False, **kwargs):
        super(CRNN_fpn, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.rnn_type = rnn_type
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN_FPN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if self.rnn_type == 'BGRU':
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
            self.rnn_2 = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
            self.rnn_4 = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)

        elif self.rnn_type =='TCN':
            # Number of [n_RNN_cell] needs to be defined
            self.rnn = TemporalConvNet(self.cnn.nb_filters[-1], [n_RNN_cell] * 2, 3, dropout=0.25)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)

        self.upsample_2 = nn.Upsample((157,1), mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample((78,1), mode='bilinear', align_corners=True)
        self.conv1x1_2 = nn.Conv2d(512,256,1) # for x_2
        self.conv1x1_4 = nn.Conv2d(512,256,1) # for x

    def forward(self, x, inference=False):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x, x_2, x_4 = self.cnn(x)
        bs, chan, frames, freq = x.size()
        bs_2, chan_2, frames_2, freq_2 = x_2.size()
        bs_4, chan_4, frames_4, freq_4 = x_4.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)
        
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)        
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
            x_2 = x_2.squeeze(-1)
            x_2 = x_2.permute(0, 2, 1)  # [bs, frames, chan]        
            x_4 = x_4.squeeze(-1)
            x_4 = x_4.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        if self.rnn_type == 'BGRU':
            x = self.rnn(x)
            x = x.permute(0, 2, 1)
            x_2 = self.rnn_2(x_2)
            x_2 = x_2.permute(0, 2, 1)  # [bs, chan, frames] 
            x_4 = self.rnn_4(x_4)
            x_4 = x_4.permute(0, 2, 1)  # [bs, chan, frames]
        elif self.rnn_type == 'TCN':
            x = self.rnn(x.transpose(1, 2)).transpose(1, 2)

        x = self.dropout(x).unsqueeze(-1)

        # feature pyramid component
        x_2 = self.dropout(x_2).unsqueeze(-1)
        x_4 = self.dropout(x_4).unsqueeze(-1)
        x_2 = torch.cat((x_2, self.upsample_4(x_4)), 1)
        x_2 = self.conv1x1_2(x_2)
        x = torch.cat((x, self.upsample_2(x_2)), 1)
        x = self.conv1x1_4(x).squeeze(-1)

        x = x.permute(0, 2, 1)
        
        # x = self.dropout(x)
        d_input = x 
             
        return x, d_input

class Predictor(nn.Module):
    def __init__(self, nclass, attention=False, n_RNN_cell=64, **kwargs):
        super(Predictor, self).__init__()
        self.attention = attention
        self.dense = nn.Linear(n_RNN_cell*2, nclass)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell*2, nclass)
            self.softmax = nn.Softmax(dim=-1) # attention over class axis, dim=-2 is attention over time axis

    
    def forward(self, x, inference=False):
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak = strong.mean(1)

        if inference:
            check = (weak > 0.5).type(torch.FloatTensor).cuda()
            check = check.unsqueeze(1).repeat(1,157,1)
            strong = strong * check
        
        
        return strong, weak

if __name__ == '__main__':
    x = torch.rand(24,1,628,128)
    nnet = CRNN(1, 10, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
            attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
            pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    encoded_x, d_input = nnet(x)
    nnet_param = [item for item in nnet.named_parameters()]

    # Clip discriminator
    clip_discriminator_kwargs = {"input_dim": 256, "dropout": 0.5}
    clip_discriminator = Clip_Discriminator(**clip_discriminator_kwargs)
    clip_domain_out = clip_discriminator(d_input)

    # Frame discriminator
    frame_discriminator_kwargs = {"input_dim": 256, "dropout": 0.5}
    frame_discriminator = Frame_Discriminator(**frame_discriminator_kwargs)
    frame_domain_out = frame_discriminator(d_input)

    # label predictor
    predictor_kwargs = {"nclass":10, "attention":True, "n_RNN_cell":128}
    predictor = Predictor(**predictor_kwargs)
    predictor_param = [item for item in predictor.named_parameters()]
    strong, weak = predictor(encoded_x)

    pdb.set_trace()

