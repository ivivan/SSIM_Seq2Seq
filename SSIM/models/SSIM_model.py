import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math, os, time

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd

from SSIM.utils.early_stopping import EarlyStopping
from SSIM.utils.VLSW import train_val_test_generate, train_test_split_SSIM, test_pm25_single_station
from SSIM.utils.support import *

# visualization
from visdom import Visdom
from torchnet import meter
from SSIM.utils.visual_loss import Visualizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########### Model ##########


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p

        self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim, hidden_size=self.enc_hid_dim, num_layers=self.enc_layers,
                            bidirectional=True)
        self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, input_len):
        embedded = self.dropout(torch.tanh(self.input_linear(input)))

        # padding variable length input
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=False,
                                                            enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        hidden = torch.tanh(self.output_linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # for different number of decoder layers
        hidden = hidden.repeat(self.dec_layers, 1, 1)

        return outputs, (hidden, hidden)


class Global_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Global_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # only pick up last layer hidden
        hidden = torch.unbind(hidden, dim=0)[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dec_layers, dropout_p, attention):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p
        self.attention = attention

        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim, hidden_size=self.dec_hid_dim,
                            num_layers=self.dec_layers)
        self.out = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        a = self.attention(hidden, encoder_outputs, mask)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted), dim=1))

        return output.squeeze(1), (hidden, cell), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src, max_len):
        mask = (src[:, :, 0] != 0).permute(1, 0)
        mask = mask[:, :max_len]
        return mask

    def forward(self, src, trg, src_len, before_sequence_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)

        max_src_len = torch.max(src_len).type(torch.int16)
        # save attn states
        decoder_attn = torch.zeros(max_len, batch_size, max_src_len).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)

        tensor_list = []

        for i, value in enumerate(before_sequence_len):
            value = value.int()
            tensor_list.append(src[value - 1, i, 0])
        output = torch.stack(tensor_list)

        mask = self.create_mask(src, max_src_len)

        for t in range(0, max_len):
            output, (hidden, cell), attn_weight = self.decoder(output, hidden, cell, encoder_outputs, mask)

            decoder_attn[t] = attn_weight

            outputs[t] = output.unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio

            output = (trg[t].view(-1) if teacher_force else output)

        # return outputs, decoder_attn
        return outputs

