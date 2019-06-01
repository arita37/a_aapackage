# -*- coding: utf-8 -*-
"""
Lightweight Functional interface to wrap access
to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic



"""
import glob
import os
import re
import argparse
from importlib import import_module

# from aapackage.mlmodel import util
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



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


def create(modelname="",  modeltype="model_dl", params=None):
    """
      modelname:  1_lstm
      modeltype:  model_dl, model_dl_tch, model_rl
      
      
    """
    modelname = modelname.replace(".py", "")
    module_path = glob.glob("{}.py".format(modelname))
    if len(module_path) == 0:
        raise NameError("Module {} notfound".format(modelname))

    module = import_module("{a}.{b}".format(a=modeltype , b=modelname))

    if params:
        model = module.Model(**params)
        return module, model
    else:
        return module, None


def load(folder, filename):
    pass


def save(model, folder, saveformat=""):
    pass


def fit(model, module, X):
    return module.fit(model, X)


def predict(model, module, sess, X):
    return module.predict(model, sess, X)


def predict_file(model, foldername=None, fileprefix=None):
    pass


def fit_file(model, foldername=None, fileprefix=None):
    pass



###############################################################################
def test_all(parent_folder="model_dl"):
    module_names = get_recursive_files(parent_folder, r"[0-9]+_.+\.py$")

    failed_scripts = []
    import tensorflow as tf

    for module_name in module_names:
        print("#######################")
        print(module_name)
        print("######################")
        module = import_module("{}.{}".format(parent_folder, module_name.replace(".py", "")))
        module.test()
        tf.reset_default_graph()
        del module




###############################################################################
def load_arguments(config_file= None ):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "config.toml")

    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default=config_file, help="Params File")
    p.add_argument("--config_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")  

    p.add_argument("--do", help="test") 
    p.add_argument("--name", help=".")  

    args = p.parse_args()
    args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args



if __name__ == "__main__":

    # test_all() # tot test all te modules inside model_dl

    args = load_arguments()

    # still not supported yet
    if args.do == "test":
        module, _ = create(args.name, None)  # '1_lstm'
        module.test()
