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



import pandas as pd
import optuna

from util import load_config


import models




def optim(modelname="model_dl.1_lstm.py", 
          pars= {},         
          optim_engine="optuna",
          optim_methhod="normal/prune",
          save_folder="/mymodel/", log_folder="") :
    """
    
   
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
    
         
         
         
        pars=  {
            "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" :(0.001, 0.1)}, 
            "num_layers": 1,
            "size_layer": {"type" : 'categorical', "value": [128, 256 ] }
            "output_size": {"type" : 'categorical', "value": [100] },
            "timestep":  {"type" : 'categorical', "value": [5] },
            "epoch":  {"type" : 'categorical', "value": [100] },
        }  
    
    
    
    
    """
    pass
    
    
    
          	
    
    
    
    
          	





#################################################################################
def get_recursive_files(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append(file)

    return outFiles



def module_load(modelname=""):
    """
      modelname:  model_dl.1_lstm.py
      model_*****/      
      
    """
    print(modelname)
    modelname = modelname.replace(".py", "")
    
    try :
      module = import_module("{a}".format(a=modelname))
    except Exception as e :
      raise NameError("Module {} notfound, {}".format(modelname, e))    
    
    return module



def create(modelname="", params=None, choice=['module', "model"]):
    """
      modelname:  model_dl.1_lstm.py
      model_*****/      
      
    """
    module = module_load(modelname=modelname)  
    model = module.Model(**params)
    return module, model
    


def create_instance_tch(name="net", params={}):
    _, model = create(name, params)
    return model   
       





def fit(model, module, X):
    return module.fit(model, X)


def predict(model, module, sess=None, X=None):
    return module.predict(model, sess, X)







#################################################################################
#################################################################################
def test_all(parent_folder="model_dl"):
    module_names = get_recursive_files(parent_folder, r"[0-9]+_.+\.py$")
    module_names.sort()
    print(module_names)
    
    failed_scripts = []
    import tensorflow as tf

    for module_name in module_names:
        print("#######################")
        print(module_name)
        print("######################")
        try :
          module = import_module("{}.{}".format(parent_folder, module_name.replace(".py", "")))
          module.test()
          tf.reset_default_graph()
          del module
        except Exception as e:
          print("Failed", e)


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



if __name__ == "__main__":
    # test_all() # tot test all te modules inside model_dl
    args = load_arguments()

    if args.do == "testall"  :
        print(args.do)
        test_all()


    if args.do == "test"  :
        print(args.do)
        module = module_load(args.modelname)  # '1_lstm'
        print(module)
        module.test()


    if args.do == "fit"  :
        module = module_load(args.modelname)  # '1_lstm'
        module.fit()
        
        
    if args.do == "predict"  :
        module = module_load(args.modelname)  # '1_lstm'
        module.predict()        


