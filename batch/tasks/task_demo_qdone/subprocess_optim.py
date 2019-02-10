#!/bin/python
"""

This is a dummy optimizer, designed to be executed in a independend python process.

Notable interactions:

loads the file "parameters.toml", located at its same folder.

outputs "results.txt" file, containing an array that can be sourced inside python.

Those interactions are defined in batch_sequencer.py and should be conserved among all optimizers.


"""
import os, sys
import pandas as pd
import arrow
import toml


from utils import logs, batch_result_folder, load_data_session, save_results, APP_ID, logs 








###########################################################################################
###########################################################################################
APP_ID   =  log_get_APPID()


def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", "--hyperparam_ii", default="0"  help="")
    options = parser.parse_args()
    return options




#############################################################################################
######## Custom Code ########################################################################
from scipy import optimize

BATCH_RESULT = batch_result_folder( "../../ztest/out/" )



def optimizerFunction(x):
    x1, x2, x3, x4 = x
    delta = x1 ** 2 * x4
    omega = -500 + delta * x1 + x2

    return omega



def execute(ii, args):
    res = optimize.minimize(optimizerFunction, [
            args["x1"],
            args["x2"],
            args["x3"],
            args["x4"]
          ])
    logs("Result: %s" % res.x)


    save_results( BATCH_RESULT, res.x,)
    logs("Finished Program: %s" % str(os.getpid()))































###########################################################################################
###########################################################################################
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    args = load_arguments()
    # ii = int(sys.argv[1])
    ii = args.hyperparams_ii


    ##### Hyper          ##########################################################  
    hyperparams = pd.read_csv("hyperparams.csv")
    arg_dict     = hyperparams.iloc[ii].to_dict()



    ##### Session data   ##########################################################
    load_data_session( arg_dict["file_data"] , method = arg_dict["file_data_method"] ) 



    execute(ii, arg_dict)




