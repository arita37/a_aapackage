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


from utils import batch_result_folder, load_data_session, save_results, OUTFOLDER, os_folder_create

from aapackage import util_log


##### Logs     ############################################################################
os_folder_create( OUTFOLDER )
APP_ID = util_log.create_appid(__file__)
LOG_FILE = os.path.join(OUTFOLDER ,"log_sub_"  + str(os.getpid()) + ".log" )

def log(s="", s1=""):
       util_log.printlog(s=s, s1=s1, app_id= APP_ID, logfile= LOG_FILE )

log("start")



###########################################################################################
###########################################################################################
def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-hy", "--hyperparam_ii", default="0", help="", type=int)
    options = parser.parse_args()
    return options







#############################################################################################
######## Custom Code ########################################################################
from scipy import optimize

#BATCH_RESULT = batch_result_folder( "../../ztest/out/" )
#LOG_FILE = "batch/ztest/log_file"  + str(os.getpid()) + ".log"
#APP_ID = util_log.create_appid(__file__)


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
    log("Result: %s" % res.x)


    save_results( OUTFOLDER, res.x, ii, arg_dict.get("file_data"))
    log("Finished Program: %s" % str(os.getpid()))









###########################################################################################
###########################################################################################
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    args = load_arguments()
    # ii = int(sys.argv[1])
    ii = args.hyperparam_ii


    ##### Hyper          ##########################################################
    hyperparams = pd.read_csv("hyperparams.csv")
    arg_dict     = hyperparams.iloc[ii].to_dict()

    log("step2")

    ##### Session data   ##########################################################
    load_data_session( arg_dict.get("file_data") , method = arg_dict.get("file_data_method") )



    execute(ii, arg_dict)




