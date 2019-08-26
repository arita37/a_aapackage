import numpy as np
import sys, os

folder_win = r"D:/_devs/Python01/gitdev/zs3drive/"
export_folder = "/home/ubuntu/proj/control/" if sys.platform != 'win32' else folder_win

export_folder += "/test_3assets/"

if not os.path.exists(export_folder):
    os.makedirs(export_folder)


class Config(object):
    n_layer = 4
    batch_size = 64
    valid_size = 256
    step_boundaries = [2000, 4000]
    num_iterations = 2000
    logging_frequency = 100
    verbose = True
    y_init_range = [0, 1]
    dilations = [1, 2, 4, 8]
    clayer = 1
    x_path = 'logs/x.npy'
    dw_path = 'logs/dw.npy'


class PricingOptionConfig(Config):
    # 6.5 option price by formulae
    dim = 3
    total_time = 3.0
    num_time_interval = 30
    num_iterations = 2000

    n_hidden_lstm = dim * 15
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    # num_hiddens = [dim, dim+10, dim+10, dim]

    num_hiddens = [dim, dim + 10, dim + 10]
    y_init_range = [1, 10]


class AllenCahnConfig(Config):
    total_time = 0.3
    num_time_interval = 10
    dim = 10
    n_hidden_lstm = 10
    lr_values = list(np.array([5e-4, 5e-4]))
    lr_boundaries = [2000]
    num_iterations = 100
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [0.3, 0.6]


class HJBConfig(Config):
    # Y_0 is about 4.5901.
    dim = 100
    total_time = 1.0
    n_hidden_lstm = 10
    num_time_interval = 20
    lr_boundaries = [400]
    num_iterations = 10000
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, dim + 50, dim + 50, dim]
    y_init_range = [0, 1]


class PricingDefaultRiskConfig(Config):
    dim = 100
    total_time = 1
    n_hidden_lstm = 100
    num_time_interval = 40
    lr_values = list(np.array([8e-3, 8e-3]))
    lr_boundaries = [3000]
    num_iterations = 6000
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [40, 50]


class BurgesTypeConfig(Config):
    dim = 50
    total_time = 0.2
    n_hidden_lstm = 100
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-3, 1e-4]))
    lr_boundaries = [15000, 25000]
    num_iterations = 30000
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [2, 4]


class QuadraticGradientsConfig(Config):
    dim = 100
    total_time = 1.0
    n_hidden_lstm = 100
    num_time_interval = 30
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [2, 4]


class ReactionDiffusionConfig(Config):
    dim = 100
    total_time = 1.0
    n_hidden_lstm = 100
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-2, 1e-2]))
    lr_boundaries = [8000, 16000]
    num_iterations = 100
    num_hiddens = [dim, dim + 10, dim + 10, dim]


def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
