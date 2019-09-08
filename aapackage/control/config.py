import numpy as np
import sys, os

folder_win = r"D:/_devs/Python01/gitdev/zs3drive/"
export_folder = "/home/ubuntu/proj/control/" if sys.platform != 'win32' else folder_win

export_folder += "/test_3assets/"

if not os.path.exists(export_folder):
    os.makedirs(export_folder)





class Config(object):
    n_layer = 4
    batch_size = 32
    valid_size = 256
    step_boundaries = [2000, 4000]
    num_iterations = 2000
    logging_frequency = 500
    verbose = True
    y_init_range = [0, 1]


    dilations = [1, 2, 4, 8]


    #Stacking
    clayer = 1



    # x_path = 'logs/x.npy'
    # dw_path = 'logs/dw.npy'




class PricingOptionConfig(Config):
    # 6.5 option price by formulae
    dim = 3
    total_time = 3.0
    num_time_interval = 30
    num_iterations = 5000


    ## FF
    num_hiddens_ff = [dim, dim + 10, dim + 10, dim]


    ### LSTM part
    n_hidden_lstm = dim * 15
    num_hiddens_lstm = [ dim *10 ]


    ### Attention LSTM Part
    n_hidden_attn = dim * 15
    num_hiddens_attn = [ dim *10 ]


    ### Dilatons model
    dilations = [1, 2, 4, 8]


    lr_values = list(np.array([5e-3, 1e-3]))
    lr_boundaries = [2000]
    num_hiddens = [dim, dim + 10, dim + 10, dim]


    y_init_range = [1, 10]



def get_config(name):
    try:
        return globals()[name + 'Config']
    except KeyError:
        raise KeyError("Config for the required  not found.")
