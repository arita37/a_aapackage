

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
conda create -n  py36f    python=3.6.7

conda install -y  tensorflow=1.9.0 keras xgboost  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel pandas        




conda install -y mkl tensorflow=1.9.0 xgboost  keras  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel        

# pip install arrow==0.10.0 attrdict==2.0.0 backports.shutil-get-terminal-size==1.0.0  github3.py==1.2.0 jwcrypto==0.6.0 kmodes==0.9 rope-py3k==0.9.4.post1 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0             

##Install TF with
conda uninstall tensorflow --force
pip install --ignore-installed --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl 


conda install dask --no-update-deps
conda install matplotlib seaborn --no-update-deps
conda install torchvision --no-update-deps





"""
#from aapackage.mlmodel import util
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from importlib import import_module  
import glob


def create(modelname="", params={}) :
    module_path = glob.glob('model_dl/{}.py'.format(modelname))
    if len(module_path)==0:
        raise NameError("Module {} notfound".format(modelname))
    
    module = import_module('model_dl.{}'.format(modelname))
    model = module.Model(**params)
    
    return module, model
    



def load(folder, filename):
    pass


def save(model, folder, saveformat=""):
    pass


def fit(model, module, X):
    return module.fit(model, X)
    

def predict(model, module, sess, X):
    return module.predict(model, sess, X)


def predict_file(model,  foldername=None, fileprefix=None):
    pass


def fit_file(model,  foldername=None, fileprefix=None):
  pass



####################################################################################
## Testing
def test_lstm() :
  df = pd.read_csv('dataset/GOOG-year.csv')
  date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
  print( df.head(5) )


  minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
  df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
  df_log = pd.DataFrame(df_log) 

  module, model =create('1_lstm',
    {'learning_rate':0.001,'num_layers':1,
     'size':df_log.shape[1],'size_layer':128,
     'output_size':df_log.shape[1],'timestep':5,'epoch':5})

  sess = fit(model, module, df_log)
  predictions = predict(model, module, sess, df_log)
  print(predictions)



if __name__ == "__main__":
   test_lstm()




