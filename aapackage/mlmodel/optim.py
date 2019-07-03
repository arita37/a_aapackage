# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap Hyper-parameter Optimization

1st engine is optuna


https://optuna.readthedocs.io/en/stable/installation.html
https://github.com/pfnet/optuna/blob/master/examples/tensorflow_estimator_simple.py
https://github.com/pfnet/optuna/tree/master/examples



###### Model standalone run
python  models.py  --modelname model_dl.1_lstm.py  --do test



###### HyperParam standalone run
python optim.py --modelname model_dl.1_lstm.py  --do test




"""
import argparse
import glob
import os
import re
from importlib import import_module
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


import optuna

from util import load_config
from models import create, module_load
###############################################################################




def optim(modelname="model_dl.1_lstm.py", 
          pars= {},      
          data_frame = None,
          optim_engine="optuna",
          optim_methhod="normal/prune",
          save_folder="/mymodel/", log_folder="") :
    """
       Interface layer to Optuna 
       for hyperparameter optimization
     
   
       return Best Parameters 
   
   
   
   weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    # Categorical parameter
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])

    # Int parameter
    num_layers = trial.suggest_int('num_layers', 1, 3)

    # Uniform parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Discrete-uniform parameter
    drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
    """
    
    if data_frame is None:
        return -1
    
    module = module_load(modelname)    

    def objective(trial):
        param_dict =  module.get_params(choice="test", ncol_input=data_frame.shape[1], ncol_output=data_frame.shape[1])
        for param in pars:
            param_value = None
            if pars[param]['type']=='log_uniform':
                param_value = trial.suggest_loguniform(param,pars[param]['range'][0], pars[param]['range'][1])
            elif pars[param]['type']=='int':
                param_value = trial.suggest_int(param,pars[param]['range'][0], pars[param]['range'][1])
            elif pars[param]['type']=='categorical':
                param_value = trial.suggest_categorical(param,pars[param]['value'])
            elif pars[param]['type']=='discrete_uniform':
                param_value = trial.suggest_discrete_uniform(param, pars[param]['init'],pars[param]['range'][0],pars[param]['range'][1])
            elif pars[param]['type']=='uniform':
                param_value = trial.suggest_uniform(param,pars[param]['range'][0], pars[param]['range'][1])
            else:
                raise Exception('Not supported type {}'.format(pars[param]['type']))

            param_dict[param] = param_value
        model = module.Model(**param_dict)
        sess = module.fit(model,data_frame)
        del sess
        return model.stats["loss"]
    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=20)  # Invoke optimization of the objective function.
    return study.best_params



###############################################################################
def load_arguments(config_file= None ):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "config.toml")
    print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")  

    p.add_argument("--do", default="test", help="test") 
    p.add_argument("--modelname", default="model_dl.1_lstm.py",  help=".")  
    p.add_argument("--dataname", default="model_dl.1_lstm.py",  help=".") 
    p.add_argument("--data", default="model_dl.1_lstm.py",  help=".")     
    
    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args

def test_all():
    df = pd.read_csv('dataset/GOOG-year.csv')
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()


    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
    df_log = pd.DataFrame(df_log) 
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)}, 
        "num_layers":    {"type": "int", "init": 2,  "range" :(2, 4)}, 
        "size_layer":    {"type" : 'categorical', "value": [128, 256 ] },
        "timestep":      {"type" : 'categorical', "value": [5] },
        "epoch":        {"type" : 'categorical', "value": [100] },
    }  
    res = optim('model_dl.1_lstm', pars=pars, data_frame = df_log)  # '1_lstm'
    print(res)

if __name__ == "__main__":
    #test_all() # tot test all te modules inside model_dl
    
    args = load_arguments()


    if args.do == "test"  :
        print(args.do)
        module = module_load(args.modelname)  # '1_lstm'
        print(module)
        module.test()


    if args.do == "search"  :
        d = json.load(args.optim_config)
        res = optim(args.modelname, d)  # '1_lstm'
        print(res)
    

        
