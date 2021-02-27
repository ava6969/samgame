import logging
from models.utils import LSTM
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from models.utils import ResNet
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import TensorType
from typing import List, Dict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsembleNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_space_ = obs_space.original_space
        data, images, privates = obs_space_.spaces['data'], obs_space_.spaces['images'], \
                                 obs_space_.spaces['privates']

        N, T, L = data.shape
        adjusted_data_shape = (T, N*L)
        _, w, h, c = images.shape
        shape = (c*N, w, h)
        self.img_shape = shape

        conv_filters = model_config.get('conv_filters')
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [100, 100])
        lstm_dim = model_config.get("lstm_cell_size", 128)

        if not conv_filters:
            conv_filters = [16, 32, 32]

        max_pool = [3] * len(conv_filters)

        conv_seqs = []

        self.lstm_net = LSTM(input_dim=adjusted_data_shape[-1], hidden_dim=lstm_dim, num_layers=2)
        for (out_channels, mp) in zip(conv_filters, max_pool):
            conv_seq = ResNet(shape, out_channels, mp)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs.append(nn.Flatten())
        self.conv_seqs = nn.ModuleList(conv_seqs)

        prev_layer_size = lstm_dim + int(np.product(privates.shape)) + int(np.product(shape))

        layers = []
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)
        self._features = None

        self._policy_net = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation)

        self._value_net = SlimFC(
                    in_size=prev_layer_size,
                    out_size=1,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict['obs']
        data, images, privates = obs['data'], obs['images'], obs['privates']
        b = privates.shape[0]
        N = data.shape[1]
        T = data.shape[2]
        # lstm
        # x1 = (td - torch.min(td)) / (torch.max(td) - torch.min(td)) normalize
        lstm_in = data.permute(0, 2, 1, 3).contiguous().view(b, T, -1)
        lstm_out = self.lstm_net(lstm_in)

        # cnn
        images = images / 255
        conv_in = images.view(b, *self.img_shape)
        for conv_seq in self.conv_seqs:
            conv_in = conv_seq(conv_in)
            if not isinstance(conv_seq, nn.Flatten):
                conv_in = F.relu(conv_in)

        x = torch.cat([privates, lstm_out, conv_in], dim=1)
        self._features = self._hidden_layers(x)
        logits = self._policy_net.forward(self._features)
        return logits, state

    def value_function(self):
        assert self._features is not None
        return self._value_net.forward(self._features).squeeze(-1)


ModelCatalog.register_custom_model("ensemble_net", EnsembleNet)
