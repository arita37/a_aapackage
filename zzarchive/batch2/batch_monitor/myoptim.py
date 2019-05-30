#  3) myscript_optim.py  ii  hyperparam_file

import os
import sys

import numpy as np

import pygmo as pyg

### relative path
DIRCWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')







args.parse()




   df = pd.csv_read( args.hyperparam_file ) 
   df = df.iloc[ii, :]  # only 1 line

   import_load(  df["myfun_file"].values )  as myfun_file # "folder/myfun.py"   string load of module.

   optim1 = optim.optim(
              fun_eval=     myfun_file.mypersoname_eval,        
              fun_bound=    myfun_file.mypersoname_bound,        
              fun_gradient= None,
              algo= "de_1227",
              algo_origin="scipy/pygmo/...",
              dict_params = df.to_dict()  
            )

   res = optim1.optimize()
   vv  = res.dump()
   util.logs( vv , type="csv")
