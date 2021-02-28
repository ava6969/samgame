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


class MLSTM_NET(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_space_ = obs_space.original_space
        data, privates = obs_space_.spaces['data'], obs_space_.spaces['privates']

        N, T, L = data.shape
        adjusted_data_shape = (T, N*L)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [100, 100])
        lstm_dim = model_config.get("lstm_cell_size", 128)

        self.lstm_net = LSTM(input_dim=adjusted_data_shape[-1], hidden_dim=lstm_dim, num_layers=2)

        prev_layer_size = lstm_dim + int(np.product(privates.shape))

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
        data, privates = obs['data'], obs['privates']
        b = privates.shape[0]
        N = data.shape[1]
        T = data.shape[2]
        # lstm
        # x1 = (td - torch.min(td)) / (torch.max(td) - torch.min(td)) normalize
        lstm_in = data.permute(0, 2, 1, 3).contiguous().view(b, T, -1)
        lstm_out = self.lstm_net(lstm_in)

        # cnn

        x = torch.cat([privates, lstm_out], dim=1)
        self._features = self._hidden_layers(x)
        logits = self._policy_net.forward(self._features)
        return logits, state

    def value_function(self):
        assert self._features is not None
        return self._value_net.forward(self._features).squeeze(-1)


ModelCatalog.register_custom_model("mlstm_net", MLSTM_NET)
