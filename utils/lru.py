import numpy as np
from typing import Optional
from functools import partial
from torch import nn
import torch


# TODO: try the parallel scan trick?

def matrix_init(shape, dtype=torch.float32, normalization=1):
    return torch.randn(shape, dtype=dtype) / normalization


def nu_init(shape, r_min, r_max, dtype=torch.float32):
    u = torch.rand(shape, dtype=dtype)
    return torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(shape, max_phase, dtype=torch.float32):
    u = torch.rand(shape, dtype=dtype)
    return torch.log(max_phase * u)


def gamma_log_init(nu, theta):
    diag_lambda = torch.exp(-torch.exp(nu) + 1j * torch.exp(theta))
    return torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))



class LRU(nn.Module):

    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    def __init__(self, d_hidden, d_model, r_min=0.0, r_max=1.0, max_phase=6.28):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.theta_log = nn.Parameter(theta_init(self.d_hidden, max_phase=self.max_phase))
        self.nu_log = nn.Parameter(
            nu_init(self.d_hidden, r_min=self.r_min, r_max=self.r_max)
        )

        # TODO: why is this a parameter?
        self.gamma_log = nn.Parameter(
            gamma_log_init(self.nu_log, self.theta_log)
        )

        # Glorot initialized Input/Output projection matrices
        self.B_re = nn.Parameter(
            matrix_init((self.d_hidden, self.d_model), normalization=np.sqrt(2 * self.d_model))
        )
        self.B_im = nn.Parameter(
            matrix_init((self.d_hidden, self.d_model), normalization=np.sqrt(2 * self.d_model))
        )
        self.C_re = nn.Parameter(
            matrix_init((self.d_model, self.d_hidden), normalization=np.sqrt(self.d_hidden))
        )
        self.C_im = nn.Parameter(
            matrix_init((self.d_model, self.d_hidden), normalization=np.sqrt(self.d_hidden))
        )
        self.D = nn.Parameter(matrix_init((self.d_model,)))


    # TODO: these operations have to be done in a batched way
    def forward(self, inputs, hidden_state):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""

        # hidden_state.shape -> [batch_size, d_hidden]
        # inputs.shape -> [seq_len, batch_size, d_model]

        # [d_hidden]
        diag_lambda = torch.complex(-torch.exp(self.nu_log), torch.exp(self.theta_log))
        diag_lambda = torch.exp(diag_lambda)
        # [d_hidden, d_model]
        B_norm = torch.complex(self.B_re, self.B_im) * torch.unsqueeze(torch.exp(self.gamma_log), dim=-1)
        # [d_model, d_hidden]
        C = torch.complex(self.C_re, self.C_im)

        outputs = torch.zeros_like(inputs)
        inputs = torch.complex(inputs, torch.zeros_like(inputs))

        # TODO: save all hidden states?
        for idx, input_ in enumerate(inputs):
            hidden_state = hidden_state * diag_lambda + input_ @ B_norm.T
            # [batch_size, d_model]
            current_outputs = (hidden_state @ C.T).real + self.D * input_.real
            outputs[idx] = current_outputs


        return outputs, hidden_state



class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    def __init__(self, d_hidden, d_model, dropout=0.0, norm="layer"):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.dropout = dropout
        self.norm = norm

        self.seq = LRU(self.d_hidden, self.d_model)
        self.out1 = nn.Linear(self.d_model, self.d_model)
        self.out2 = nn.Linear(self.d_model, self.d_model)

        # TODO: verify this
        if self.norm in ["layer"]:
            self.normalization = nn.LayerNorm((self.d_model))
        else:
            self.normalization = nn.BatchNorm()

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        else:
            self.drop = nn.Identity()


        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, hidden_state):
        # TODO: figure out normalization for complex numbers
        x = self.normalization(inputs)  # pre normalization
        x, hidden_state = self.seq(x, hidden_state)  # call LRU
        # x -> [batch_size, seq_len, d_hidden]
        x = self.drop(self.gelu(x))
        x = self.out1(x) * self.sigmoid(self.out2(x))  # GLU
        # TODO: why double dropout?
        x = self.drop(x)
        return inputs + x, hidden_state  # skip connection


class StackedLRUModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    def __init__(self, input_size, hidden_size, num_layers=4, dropout=0.0, norm="layer",
                 batch_first: bool = False, bias: bool = True):
        super().__init__()

        assert batch_first is False
        assert bias is True

        self.d_hidden = hidden_size
        self.d_model = input_size
        self.num_layers: int = num_layers
        self.dropout = dropout
        self.norm = norm

        layers = [
            SequenceLayer(
                d_hidden=self.d_hidden,
                d_model=self.d_model,
                dropout=self.dropout,
                norm=self.norm,
            )
            for _ in range(self.num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, hidden_states: Optional[torch.Tensor] = None):
        if hidden_states is None:
            hidden_states = torch.zeros(
                    (self.num_layers, x.shape[1], self.d_hidden),
                    dtype=torch.complex64, device=x.device)


        # x.shape -> [seq_len, batch_size, d_model]
        # hidden_state.shape -> [num_layers, batch_size, d_hidden]

        # TODO: make sure that shapes are correct
        new_hidden_states = torch.empty_like(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            hidden_state = hidden_states[layer_idx]
            x, new_hidden_state = layer(x, hidden_state)  # apply each layer
            new_hidden_states[layer_idx] = new_hidden_state

        return x, new_hidden_states
