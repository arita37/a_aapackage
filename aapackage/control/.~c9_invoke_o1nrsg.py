"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).


Global LSTM, hidden state share
at each time steps Input hidden --> 

Output of LSTM goes to FeedFoward local to time step ti, 
to Collapse into y function







"""

import json
import logging
import os
from argparse import ArgumentParser


import numpy as np
####################################################################################################



####################################################################################################
def load_argument() :
   p = ArgumentParser()
   p.add_argument("--problem_name", type=str, default='HJB')
   p.add_argument("--num_run", type=int, default=1)
   p.add_argument("--log_dir", type=str, default='./logs')
   p.add_argument("--framework", type=str, default='tch')
   arg = p.parse_args()
   return arg

def log(s):
    logging.info(s)


def log_init(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    
    

def config_dump(conf, path_prefix):
    with open(path_prefix + ".json", "w") as outfile:
        json.dump(
            dict( (name, getattr(conf, name)) for name in dir(conf) if not name.startswith("__")),
            outfile, indent=2,
        )    
    

def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")



import numpy as np


class Config(object):
    n_layer = 4
    batch_size = 64
    valid_size = 256
    step_boundaries = [2000, 4000]
    num_iterations = 6000
    logging_frequency = 10
    verbose = True
    y_init_range = [0, 1]


class HJBConfig(Config):
    # Y_0 is about 4.5901.
    dim = 2
    total_time = 1.0
    num_time_interval = 10
    lr_boundaries = [400]
    num_iterations = 20
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]






import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class HJB(Equation):
    """
      x_sample and dw are MultiVariate Time Series
    
      x_sample(Nsample, Nb_timeSeries, timePeriod)
      
       nsample x VecDim x dT
       
      
    
    """
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lambda = 1.0

    def sample(self, num_sample):

        ### All the samples
        dw_sample = (
            normal.rvs(size=[num_sample, self._dim, self._num_time_interval]) * self._sqrt_delta_t
        )

        ### Euler Discretization
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init

        ### Euler Sequential ; Diffusion process
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        # Constraints 1
        return -self._lambda * tf.reduce_sum(tf.square(z), 1, keepdims=True)

    def g_tf(self, t, x):
        # Constraints 2
        return tf.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from time import time


TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


def tch_to_device():
   torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    



"""
 for t in range(0, T) :
   pred_t = LSTM(X[t].  hiddenstate)
   zt = FeedFoward(t,  pred_t)
   yt =  Euler(xt, dWt, zt)  # Graident




"""

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

        #Backward
        for t in range(0, self._num_time_interval - 1):
            # print('y qian', y.max())
            y = y - self._bsde.delta_t * (self._bsde.f_th(time_stamp[t], x[:, :, t], y, z))
            # print('y hou', y.max())
            
            add = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            # print('add', add.max())
            
            y = y + add
            z = self._subnetworkList[t](x[:, :, t + 1]) / self._dim
            # print('z value', z.max())
            
        # terminal time
        y = y - self._bsde.delta_t * self._bsde.f_th(  time_stamp[-1], x[:, :, -2], y, z) \
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)

        # Error at terminal Value
        delta = y - self._bsde.g_th(self._total_time, x[:, :, -1])

        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, delta ** 2,
                                      2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))
        return loss, self._y_init



def train(config, bsde):
    # build and train
    #args = parser.parse_args()
    device = tch_to_device()
    print("Training on:", device)
    #config = get_config(args.name)
    #bsde = get_equation(args.name, config.dim, config.total_time, config.num_time_interval)

    net = FeedForwardModel(config, bsde)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), config.lr_values[0])
    t0 = time()
    
    training_history = []  # to save iteration results
    dw_valid, x_valid = bsde.sample(config.valid_size)      # for validation accuracy

    # begin sgd iteration
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            ### Accuracy on Volidation set
            net.eval()
            loss, init = net(x_valid.to(device), dw_valid.to(device))
            training_history.append([step, loss, init.item(), time() - t0])
            print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    step, loss, init.item(), time() - t0))

        ### MC sample
        dw_train, x_train = bsde.sample(config.batch_size)
        
        ### Forward compute
        optimizer.zero_grad()
        net.train()
        
        ### Backward Loss with gradient
        loss, _ = net(x_train.to(device), dw_train.to(device))
        loss.backward()
        optimizer.step()

    return np.array(training_history)
    
    
    



def main():
    arg = load_argument() 
    print(arg)
    c = get_config(arg.problem_name)


    log_init(arg.log_dir)
    path_prefix = os.path.join(arg.log_dir, arg.problem_name)
    config_dump(c, path_prefix)     


    if arg.framework == 'tf':
        import tensorflow as tf
        from equation import get_equation as get_equation_tf 
        from solver import FeedForwardModel as FFtf

        bsde = get_equation_tf(arg.problem_name, c.dim, c.total_time, c.num_time_interval)


    elif arg.framework == 'tch':
        from equation_tch import get_equation as get_equation_tch
        from solver_tch import train

        bsde = get_equation_tch(arg.problem_name, c.dim, c.total_time, c.num_time_interval)


    print("Running ", arg.problem_name, " on: ", arg.framework)
    print(bsde)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")
    
    #### Loop over run
    for k in range(1, arg.num_run + 1):
        log("Begin to solve %s with run %d" % (arg.problem_name, k))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if arg.framework == 'tf':
            tf.reset_default_graph()
            with tf.Session() as sess:
                model = FFtf(c, bsde, sess)
                model.build()
                training_history = model.train()

        elif arg.framework == 'tch':
            print("ok")
            training_history = train(c, bsde)

        if bsde.y_init:
            log("% error of Y0: %s{:.2%}".format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init),)

        # save training history
        np.savetxt(
          "{}_training_history_{}.csv".format(path_prefix, k),
          training_history,
          fmt=["%d", "%.5e", "%.5e", "%d"],
          delimiter=",",
          header="step,loss_function,target_value,elapsed_time",
          comments="",
        )






if __name__ == "__main__":
    main()
















