

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



############### conda DL ####################################################
conda create -n  py36c    python=3.6.7

conda install -y mkl tensorflow=1.9.0 xgboost  keras  lightgbm catboost pytorch scikit-learn  optuna  chainer  dask  ipykernel        

# pip install arrow==0.10.0 attrdict==2.0.0 backports.shutil-get-terminal-size==1.0.0  github3.py==1.2.0 jwcrypto==0.6.0 kmodes==0.9 rope-py3k==0.9.4.post1 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0             

##Install TF with
conda uninstall tensorflow --force
pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl 


conda install dask --no-update-deps
conda install matplotlib seaborn --no-update-deps
conda install torchvision --no-update-deps





"""
from aapackage.mlmodel import util




def create(modelname="", params={}) :
    pass



def load(folder, filename):
    pass


def save(model, folder, saveformat=""):
    pass


def fit(model, X=None, foldername=None):
  pass
    

def predict(model, X=None):
    pass


def predict_file(model,  foldername=None, fileprefix=None, output_folder):
    pass


def fit_file(model,  foldername=None, fileprefix=None):
  pass





