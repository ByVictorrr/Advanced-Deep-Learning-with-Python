import torch
import torch.nn as nn
import torch.optim as optim
import math
import typing


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, lr=0.01):
        """Init the LSTMCell.
        :param input_size: input vector size
        :param hidden_size: cell state vector size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_fc = nn.Linear(input_size, 4 * hidden_size)
        self.h_fc = nn.Linear(hidden_size, 4 * hidden_size)
        self.reset_parameters()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def reset_parameters(self):
        """Xavier initialization"""
        size = math.sqrt(3.0 / self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-size, size)

    def forward(self, x_t: torch.Tensor, hidden: typing.Tuple[torch.Tensor, torch.Tensor]) -> typing.Tuple[
        torch.Tensor, torch.Tensor]:
        h_t_1, c_t_1 = hidden
        x_t = x_t.view(-1, x_t.size(1))
        h_t_1 = h_t_1.view(-1, h_t_1.size(1))
        c_t_1 = c_t_1.view(-1, c_t_1.size(1))

        gates = self.x_fc(x_t) + self.h_fc(h_t_1)
        i_t, f_t, candidate_c_t, o_t = gates.chunk(4, 1)
        i_t, f_t, candidate_c_t, o_t = i_t.sigmoid(), f_t.sigmoid(), candidate_c_t.tanh(), o_t.sigmoid()

        c_t = torch.mul(f_t, c_t_1) + torch.mul(i_t, candidate_c_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t

