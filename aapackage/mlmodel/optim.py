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
          df = None,
          optim_engine="optuna",
          optim_method="normal/prune",
          save_folder="/mymodel/", log_folder="",ntrials=2) :

          if optim_engine == "optuna" :
            return optim_optuna(modelname, 
                pars,      
                    df,
                optim_method,
                save_folder, log_folder,ntrials) 
          return None
                    

def optim_optuna(modelname="model_dl.1_lstm.py", 
          pars= {},      
          df = None,
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
    
    if df is None:
        return -1
    
    module = module_load(modelname)    

    def objective(trial):
        param_dict =  module.get_params(choice="test", ncol_input=df.shape[1], ncol_output=df.shape[1])
        for t in pars:
            p = None
            x =pars[t]['type']
            if x=='log_uniform':
                p = trial.suggest_loguniform(t,pars[t]['range'][0], pars[t]['range'][1])
                
            elif x=='int':
                p = trial.suggest_int(t,pars[t]['range'][0], pars[t]['range'][1])
                
            elif x=='categorical':
                p = trial.suggest_categorical(t,pars[t]['value'])
                
            elif x=='discrete_uniform':
                p = trial.suggest_discrete_uniform(t, pars[t]['init'],pars[t]['range'][0],pars[t]['range'][1])
            
            elif x=='uniform':
                p = trial.suggest_uniform(t,pars[t]['range'][0], pars[t]['range'][1])
            
            else:
                raise Exception('Not supported type {}'.format(pars[t]['type']))

            param_dict[t] = p
            
        model = module.Model(**param_dict)
        sess = module.fit(model,df)
        stats = model.stats["loss"]
        del sess
        del model
        tf.reset_default_graph()
        return stats
    if optim_method=='prune':
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
    param_dict =  study.best_params
    param_dict.update(module.get_params(choice="test", ncol_input=df.shape[1], 
                                        ncol_output=df.shape[1]))
    

    model = module.Model(**param_dict)
    sess = module.fit(model,df)
    modelname = modelname.split('.')[-2] # this is the module name which contains .
    if not(os.path.isdir(save_folder)):
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


def data_loader(file_name='dataset/GOOG-year.csv'):
    df = pd.read_csv(file_name)
    
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()


    minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
    df_log = pd.DataFrame(df_log) 
    return df_log

    
    
def test_all():
    df_log = data_loader()
    pars =  {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)}, 
        "num_layers":    {"type": "int", "init": 2,  "range" :(2, 4)}, 
        "size_layer":    {"type" : 'categorical', "value": [128, 256 ] },
        "timestep":      {"type" : 'categorical', "value": [5] },
        "epoch":        {"type" : 'categorical', "value": [100] },
    }  
    res = optim('model_dl.1_lstm', pars=pars, df = df_log,ntrials=1)  # '1_lstm'
    print(res)




if __name__ == "__main__":
    #test_all() # tot test all te modules inside model_dl
    args = load_arguments()


    if args.do == "test"  :
        test_all()


    if args.do == "search"  :
        df_log = data_loader(args.data_path)
        d = json.load(open(args.config_file,'r'))
        res = optim(args.modelname, d, 
                    ntrials=int(args.ntrials), 
                    optim_engine=args.optim_engine, 
                    optim_method=args.optim_method, 
                    df=df_log, 
                    save_folder=args.save_folder)  # '1_lstm'
        print(res)
    

        
