"""
Lightweight Functional interface to wrap access
to Deep Learning, RLearning models.
Logic follows Scikit Learn API and simple for easy extentions.Logic


1) Installation as follow
   source activate yourCOndaEnv
   cd /home/ubuntu/aagit/aapackage/
   pip install -e .

This will install editable package, and this can be used
   from aapackage.mlmodel import models

2) All code and data are in this folder
  /home/ubuntu/aagit/aapackage/mlmodel/


############### conda DL #######################################################
conda create -n  py36f    python=3.6.7

conda install -y  tensorflow=1.9.0 keras xgboost  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel pandas        
conda install matplotlib seaborn --no-update-deps


##Install TF with AVC
conda uninstall tensorflow --force
anaconda3/envs/py36d/bin/pip install --ignore-installed --upgrade  https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl  




##############################################################################
conda create -n  py36e    python=3.6.7

conda install -y mkl tensorflow=1.9.0 xgboost  keras  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel        

# pip install arrow==0.10.0 attrdict==2.0.0 backports.shutil-get-terminal-size==1.0.0  github3.py==1.2.0 jwcrypto==0.6.0 kmodes==0.9 rope-py3k==0.9.4.post1 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0             

##Install TF with
conda uninstall tensorflow --force

anaconda3/envs/py36d/bin/pip install --ignore-installed --upgrade  https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl  



anaconda3/envs/py36c/bin/pip install -e /home/ubuntu/aagit/aapackage/





pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl 

pip install --ignore-installed --upgrade  https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl  --user



conda install dask --no-update-deps

conda install torchvision --no-update-deps




anaconda3/envs/py36c/bin/pip 


"""
import glob
import os
import re
from importlib import import_module

# from aapackage.mlmodel import util
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_recursive_files(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append(file)

    return outFiles


def create(modelname="", params=None):
    """
      modelname= model_dl/1_lstm
      
      
    """
    modelname = modelname.replace(".py", "")
    module_path = glob.glob("{}.py".format(modelname))
    if len(module_path) == 0:
        raise NameError("Module {} notfound".format(modelname))

    module = import_module("model_dl.{}".format(modelname))

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


########################################################################
##### CLI1
def load_arguments():
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, "config.toml")

    p = argparse.ArgumentParser()
    p.add_argument("--param_file", default=config_file, help="Params File")
    p.add_argument("--param_mode", default="test", help="test/ prod /uat")
    p.add_argument("--log_file", help="log.log")  # default="batchdaemon_autoscale.log",

    p.add_argument("--do", help="test")  # default="nodaemon",
    p.add_argument("--name", help=".")  # default="batchdaemon_autoscale.log",

    args = p.parse_args()

    ##### Load file params as dict namespace #########################
    class to_namespace(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    import toml

    try:
        pars = toml.load(args.param_file)
        pars = pars[args.param_mode]  # test / prod

        ### Overwrite params by CLI input and merge with toml file
        for key, x in vars(args).items():
            if x is not None:  # only values NOT set by CLI
                pars[key] = x

        print(pars)
        pars = to_namespace(pars)  #  like object/namespace pars.instance
        return pars

    except:
        return args


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


if __name__ == "__main__":

    # test_all() # tot test all te modules inside model_dl

    args = load_arguments()

    # still not supported yet
    if args.do == "test":
        module, _ = create(args.name, None)  # '1_lstm'
        module.test()
