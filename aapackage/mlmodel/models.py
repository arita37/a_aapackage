# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic


from models import create
model = create_instance("model_tf.1_lstm.py", dict_params=params_dict)  # Net



### RL model
python  models.py  --modelname model_rl.4_policygradient  --do test


### TF DNN model
python  models.py  --modelname model_tf.1_lstm.py  --do test


## PyTorch models
python  models.py  --modelname model_tch.mlp.py  --do test



"""
import argparse
import glob
import os
import re
from importlib import import_module

# from aapackage.mlmodel import util
import pandas as pd
import tensorflow as tf



#################################################################################
from util import load_config, get_recursive_files




#################################################################################
def module_load(modelname=""):
    """
      Load the file which contains the model description
      modelname:  model_tf.1_lstm.py
      model_*****/      
      
    """
    print(modelname)
    modelname = modelname.replace(".py", "")
    
    try :
      module = import_module("{a}".format(a=modelname))
    except Exception as e :
      raise NameError("Module {} notfound, {}".format(modelname, e))    
    
    return module



def create(modelname="", dict_params=None, choice=['module', "model"]):
    """
      Create Instance of the model
      modelname:  model_tf.1_lstm.py
      model_*****/
      dict_params : dict paras
      
    """
    module = module_load(modelname=modelname)  
    model = module.Model(**dict_params)
    return module, model


def fit(model, module, X, **kwargs):
    return module.fit(model, X, **kwargs)


def predict(model, module, sess, X, **kwargs):
    return module.predict(model, sess, X, **kwargs)


def load(folder_name, model_type="", filename=None **kwargs):
    if model_type == "tf" :
        return load_tf(folder_name)
    
    if model_type == "tch" :
        return load_tch(folder_name)
    
    if model_type == "pkl" :
        return load_pkl(folder_name, filename)


def save(folder_name, model_type="", filename=None ** kwargs):
    if model_type == "tf":
        return  1
    if model_type == "tch":
        return 1
    
    if model_type == "pkl":
        return 1



####################################################################################################
def create_instance_tch(name="net", params={}):
    _, model = create(name, params)
    return model


def load_tf(filename):
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, filename)
    return sess


def load_tch(filename):
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, filename)
    return sess


def load_pkl(folder_name) :
    pass



def save_tf(sess, file_path):
    saver = tf.train.Saver()
    return saver.save(sess, file_path)
    

def predict_file(model, foldername=None, fileprefix=None):
    pass


def fit_file(model, foldername=None, fileprefix=None):
    pass




####################################################################################################
####################################################################################################
def test_all(parent_folder="model_tf"):
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
          del module
        except Exception as e:
          print("Failed", e)


####################################################################################################
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
    p.add_argument("--modelname", default="model_tf.1_lstm.py",  help=".")
    p.add_argument("--dataname", default="model_tf.1_lstm.py",  help=".")
    p.add_argument("--data", default="model_tf.1_lstm.py",  help=".")
    
    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args



if __name__ == "__main__":
    # test_all() # tot test all te modules inside model_tf
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


