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
import tensorflow as tf


import optuna

from util import load_config
from models import create, module_load, save
###############################################################################




def optim(modelname="model_dl.1_lstm.py", 
          pars= {},      
          data_frame = None,
          optim_engine="optuna",
          optim_method="normal/prune",
          save_folder="/mymodel/", log_folder="",ntrials=2) :
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
        stats = model.stats["loss"]
        del sess
        del model
        tf.reset_default_graph()
        return stats
        
    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
    param_dict =  study.best_params
    param_dict.update(module.get_params(choice="test", ncol_input=data_frame.shape[1], ncol_output=data_frame.shape[1]))
    

    model = module.Model(**param_dict)
    sess = module.fit(model,data_frame)
    modelname = modelname.split('.')[-2] # this is the module name which contains .
    os.makedirs(save_folder)
    file_path = os.path.join(save_folder,modelname+'.ckpt')

    save(sess,file_path)

    return param_dict



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
    p.add_argument("--log_file", help="File to save the logging")  

    p.add_argument("--do", default="test", help="what to do test or search") 
    p.add_argument("--ntrials", default=100, help='number of trials during the hyperparameters tuning')
    p.add_argument("--modelname", default="model_dl.1_lstm.py",  help="name of the model to be tuned this name will be used to save the model")  
    p.add_argument("--data_path", default="dataset/GOOG-year.csv",  help="path of the training file")  
    p.add_argument('--optim_engine', default='optuna',help='Optimization engine') 
    p.add_argument('--optim_method', default='normal/prune',help='Optimization method')  
    p.add_argument('--save_folder', default='save_dir',help='folder that will contain saved version of best model')  
    
    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args

def preprocess_dataframe(file_name='dataset/GOOG-year.csv'):
    df = pd.read_csv(file_name)
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()


    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
    df_log = pd.DataFrame(df_log) 
    return df_log
    
def test_all():
    df_log = preprocess_dataframe()
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)}, 
        "num_layers":    {"type": "int", "init": 2,  "range" :(2, 4)}, 
        "size_layer":    {"type" : 'categorical', "value": [128, 256 ] },
        "timestep":      {"type" : 'categorical', "value": [5] },
        "epoch":        {"type" : 'categorical', "value": [100] },
    }  
    res = optim('model_dl.1_lstm', pars=pars, data_frame = df_log,ntrials=1)  # '1_lstm'
    print(res)

if __name__ == "__main__":
    #test_all() # tot test all te modules inside model_dl
    
    args = load_arguments()


    if args.do == "test"  :
        test_all()


    if args.do == "search"  :
        df_log = preprocess_dataframe(args.data_path)
        d = json.load(open(args.config_file,'r'))
        res = optim(args.modelname, d,ntrials=int(args.ntrials),optim_engine=args.optim_engine,optim_method=args.optim_method, data_frame=df_log, save_folder=args.save_folder)  # '1_lstm'
        print(res)
    

        
