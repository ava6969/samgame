import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ResNet(nn.Module):
    def __init__(self, input_shape, out_channels, max_pool=None):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.max_pool = None if not max_pool else nn.MaxPool2d(max_pool, max_pool)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = self.max_pool(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        in_net = self.max_pool if self.max_pool else self.conv
        if self.max_pool:
            h_ = (((h + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
            w_ = (((w + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
        else:
            h_ = (((h + 2 * in_net.padding[0] - in_net.dilation[0] *
                    (in_net.kernel_size[0] - 1) - 1) / in_net.stride[0]) + 1)
            w_ = (((w + 2 * in_net.padding[1] - in_net.dilation[1] *
                    (in_net.kernel_size[1] - 1) - 1) / in_net.stride[1]) + 1)
        return self._out_channels, int(h_), int(w_)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_out=0.3):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=drop_out)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=True).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _= self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        last_hid = out[:, -1, :]
        # out.size() --> 100, 10
        return last_hid