# -*- coding: utf-8 -*-
import os, sys
from time import time
import copy
import gc
import random
import copy
import numpy as np,  pandas as pd
from collections import OrderedDict




#####################################################################################
dirsession = r"C:/a/session/"
dirmodel = r"C:/model/"

print('os.getcwd', os.getcwd())
print('dirsession', dirsession)



#####################################################################################
def toint(x):
   try :
       return int(x)
   except :
       return x


def toint2(x, def_val=-1):
   try :
      return int(x)
   except :
      return def_val

def to_dateint(x) :
  try :  
      v = x.split("/")
      return   int( v[0] )*10000 +  int( v[1] )*100 + int( v[2] )  
  except :
      return -1


#####################################################################################
def save(x, folder, filename=None ) :
    import pickle
    fname = folder  + f"/{filename}.pkl"
    with open( fname, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
    return fname 


def load(filename=None ) :
  import pickle
  with open( f"{filename}.pkl" , 'rb') as f:
      return pickle.load(f)
     

def save_model(model, dfeatures, data, folder="", filename="model", glob=None ) :
   d = {'model'   : model,
        'features': dfeatures,
        'data'    : data }

   path = f"{folder}/{filename}.pkl"
   save( d , path )
   print( path )


def save_session(folder , glob ) :    
  if not os.path.exists(folder) :
    os.makedirs( folder )   
  
  lcheck = [ "<class 'pandas.core.frame.DataFrame'>", "<class 'list'>", "<class 'dict'>",
             "<class 'str'>" ,  "<class 'numpy.ndarray'>",
             "<class 'sklearn", 
           ]  

  lexclude = {   "In", "Out" }
  
  for x, _ in glob.items() :
     if not x.startswith('_') and  x not in lexclude :
        x_type =  str(type(glob.get(x) ))
        if x_type in lcheck or x.startswith('clf')  :
            try :
              print( save( glob[x], folder  + "/" + x + ".pkl") )
            except Exception as e:
              print(x, x_type, e)


def show_session(folder) :
  """Show Load session      
  """      
  for dirpath, subdirs, files in os.walk( folder ):
     for x in subdirs :
         print(x)


def load_session(folder, glob=None) :
  """Data Load session      
  """      
  for dirpath, subdirs, files in os.walk( folder ):
    for x in files:
       filename = os.path.join(dirpath, x) 
       x = x.replace(".pkl", "")
       try :
         glob[x] = load(  filename )
         print(filename) 
       except Exception as e :
         print(filename, e)









