# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic


from models import create
model = create_instance("model_dl.1_lstm.py", params=params_dict)  # Net



### RL model
python  models.py  --modelname model_rl.4_policygradient  --do test


### TF DNN model
python  models.py  --modelname model_dl.1_lstm.py  --do test


## PyTorch models
python  models.py  --modelname model_dl_tch.mlp.py  --do test



"""
import argparse
import glob
import os
import re
from importlib import import_module

# from aapackage.mlmodel import util
import pandas as pd

from util import load_config


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




def load(folder, filename):
    pass


def save(model, folder, saveformat=""):
    pass




def predict_file(model, foldername=None, fileprefix=None):
    pass


def fit_file(model, foldername=None, fileprefix=None):
    pass




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


