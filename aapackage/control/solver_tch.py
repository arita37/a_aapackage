import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from config import get_config
from equation_tch import get_equation

from argparse import ArgumentParser

import time


TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

parser = ArgumentParser()
parser.add_argument("--name", type=str, default='ReactionDiffusion')


class Dense(nn.Module):

    def __init__(self, cin, cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = nn.BatchNorm1d(cout, eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        nn.init.normal_(self.linear.weight, std=5.0 / np.sqrt(cin + cout))

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = F.relu(x)
        return x


class Subnetwork(nn.Module):

    def __init__(self, config):
        super(Subnetwork, self).__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim, eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config.num_hiddens[i - 1], config.num_hiddens[i]) for i in
                       range(1, len(config.num_hiddens) - 1)]
        self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""

    def __init__(self, config, bsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self._bsde = bsde

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        self._y_init = Parameter(torch.Tensor([1]))
        self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])
        self._subnetworkList = nn.ModuleList([Subnetwork(config) for _ in range(self._num_time_interval - 1)])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x, dw):
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        z_init = torch.zeros([1, self._dim]).uniform_(-.1, .1).to(TH_DTYPE).to(self.device)

        all_one_vec = torch.ones((dw.shape[0], 1), dtype=TH_DTYPE).to(self.device)
        y = all_one_vec * self._y_init

        z = torch.matmul(all_one_vec, z_init)

        for t in range(0, self._num_time_interval - 1):
            # print('y qian', y.max())
            y = y - self._bsde.delta_t * (
                self._bsde.f_th(time_stamp[t], x[:, :, t], y, z)
            )
            # print('y hou', y.max())
            add = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            # print('add', add.max())
            y = y + add
            z = self._subnetworkList[t](x[:, :, t + 1]) / self._dim
            # print('z value', z.max())
        # terminal time
        y = y - self._bsde.delta_t * self._bsde.f_th( \
            time_stamp[-1], x[:, :, -2], y, z \
            ) + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)

        delta = y - self._bsde.g_th(self._total_time, x[:, :, -1])

        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta ** 2,
                                      2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))
        return loss, self._y_init


def train():
    # build and train
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Training on:", device)
    config = get_config(args.name)
    bsde = get_equation(args.name, config.dim, config.total_time, config.num_time_interval)

    net = FeedForwardModel(config, bsde)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), config.lr_values[0])
    start_time = time.time()
    # to save iteration results
    training_history = []
    # for validation
    dw_valid, x_valid = bsde.sample(config.valid_size)

    # begin sgd iteration
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.eval()
            loss, init = net(x_valid.to(device), dw_valid.to(device))

            elapsed_time = time.time() - start_time
            training_history.append([step, loss, init.item(), elapsed_time])
            if config.verbose:
                print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    step, loss, init.item(), elapsed_time))

        dw_train, x_train = bsde.sample(config.batch_size)
        optimizer.zero_grad()
        net.train()
        loss, _ = net(x_train.to(device), dw_train.to(device))
        loss.backward()

        optimizer.step()


if __name__ == '__main__':
    train()