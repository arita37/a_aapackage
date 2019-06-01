    
from __future__ import print_function
import argparse
import os
import toml




#####################################################################################
def load_config(args, config_file, config_mode, verbose=0) :
    ##### Load file params as dict namespace #########################
    import toml
    class to_namespace(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    if verbose :
      print(config_file)
    try :
      pars = toml.load(config_file)
      # print(args.param_file, pars)
    except Exception as e:
        print(e)
        return args
        
    pars = pars[config_mode]  # test / prod
    if verbose :
        print(config_file, pars)

    ### Overwrite params from CLI input and merge with toml file
    for key, x in vars(args).items():
        if x is not None:  # only values NOT set by CLI
            pars[key] = x

    # print(pars)
    pars = to_namespace(pars)  #  like object/namespace pars.instance
    return pars
    