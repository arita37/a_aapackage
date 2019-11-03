# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic


######### Code sample  ##########################################################
from models import create
# module, model = create_full("model_tf.1_lstm.py", dict_params= model_params)  # Net

model_params =  { "learning_rate": 0.001, "num_layers": 1,
            "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,  "epoch": 2,}
data_params = {}

module = module_load(modelname="model_tf.1_lstm.py")
model = module.create(module, model_params)

df = data_loader(data_params)
sess = module.fit(model, df)
stats = model.stats["loss"]
module.save("myfolder/")



######### Command line sample  #################################################
### RL model
python  models.py  --modelname model_tf.rl.4_policygradient  --do test


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
# import pandas as pd
# import tensorflow as tf



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
      module = import_module(f"{modelname}")
    except Exception as e :
      raise NameError("Module {} notfound, {}".format(modelname, e))

    return module


def create(module, model_params=None):
    """
      Create Instance of the model from loaded module model
      module *****/myfile.py
      model_params : dict paras

    """
    if model_params is None :
      model_params = module.get_params()

    model = module.Model(**model_params)
    return model


def create_full(modelname="", model_params=None, choice=['module', "model"]):
    """
      Create Instance of the model
      modelname:  model_tf.1_lstm.py
      dict_params : dict paras

    """
    module = module_load(modelname=modelname)
    model = module.Model(**model_params)
    return module, model


def fit(model, module, data_params, **kwargs):
    return module.fit(model, data_params, **kwargs)


def predict(model, module, sess, data_params, **kwargs):
    return module.predict(model, sess, data_params, **kwargs)


def load(folder_name, model_type="tf", filename=None, **kwargs):
    if model_type == "tf" :
        return load_tf(folder_name)

    if model_type == "tch" :
        return load_tch(folder_name)

    if model_type == "pkl" :
        return load_pkl(folder_name)


def save(folder_name, modelname=None, model_type="tf", ** kwargs):
    if model_type == "tf":
      #save_folder = save_folder + "/" + modelname
      if not(os.path.isdir(folder_name)):
        os.makedirs(folder_name)
      file_path = f"{folder_name}/{modelname}.ckpt"
      save_tf(sess, file_path)
      print(file_path)


    if model_type == "tch":
        return 1

    if model_type == "pkl":
        return 1



########## TF specific ################################################################################
def load_tf(filename):
    return sess


def save_tf(sess, file_path):
    saver = tf.train.Saver()
    return saver.save(sess, file_path)


########## pyTorch specific ##########################################################################
def create_instance_tch(name="net", params={}):
    _, model = create(name, params)
    return model

def load_tch(filename):
    return sess




########## Other model specific #####################################################################
def load_pkl(folder_name, filename=None) :
    pass

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

    if parent_folder == "model_tf" :
       import tensorflow as tf

    for module_name in module_names:
        print("#######################")
        print(module_name)
        print("#######################")
        try :
          module = import_module(f'{parent_folder}.{module_name.replace(".py", "")}')
          module.test()
          del module
        except Exception as e:
          print("Failed", e)



####################################################################################################
############CLI Command ############################################################################
def load_arguments(config_file= None ):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "models_config.json")
    print(config_file)

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")

    p.add_argument("--do", default="test", help="test")
    p.add_argument("--modelname", default="model_tf.1_lstm.py",  help=".")
    p.add_argument("--dataname", default="dataset/google.csv",  help=".")
                                 
    p.add_argument("--save_folder", default="ztest/",  help=".")
    p.add_argument("--load_folder", default="ztest/",  help=".")
                                 
    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args


                                 
def get_params(arg) :    
   js = json.load(open(arg.config_file, 'r'))  #Config     
   js = js[arg.config_mode]  #test /uat /prod                              
   model_params = js.get("model_params")                          
   data_params = js.get("data_params")  
                                                              
   return model_params, data_params                              
                                 
                                 
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
        model_params, data_params = get_params(args)                          
                                 
        module = module_load(args.modelname)  # '1_lstm'
        model = module.create(**model_params)                         
        sess = module.fit(model, data_params)
        save(sess, args.save_folder)

                                 
    if args.do == "predict"  :
        model_params, data_params = get_params(args)      
                                 
        module = module_load(args.modelname)  # '1_lstm'
        model = load(folder_name)                         
        module.predict(model, data_params, data_params["target_folder"] )





