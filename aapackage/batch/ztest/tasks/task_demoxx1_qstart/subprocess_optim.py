#!/bin/python
"""
This is a dummy optimizer, designed to be executed in a independend python process.

Notable interactions:

loads the file "parameters.toml", located at its same folder.


"""
import os
import sys

import arrow
import pandas as pd
import toml
#########################################################################################
######## Custom Code ####################################################################
from scipy import optimize

from aapackage import util_log
from utils import OUTFOLDER, load_data_session, os_chdir, os_folder_create, save_results

##### Logs     ##########################################################################
print(os.getcwd())
print("outfolder", OUTFOLDER)
os_folder_create(OUTFOLDER)
APP_ID = util_log.create_appid(__file__)
LOG_FILE = os.path.join(OUTFOLDER, "log_sub_" + str(os.getpid()) + ".log")


def log(s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9="", s10=""):
    util_log.printlog(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, app_id=APP_ID, logfile=LOG_FILE)


log("start")


#########################################################################################
def load_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-hy", "--hyperparam_ii", default="0", help="", type=int)
    options = parser.parse_args()
    return options


def optimizerFunction(x):
    x1, x2, x3, x4 = x
    delta = x1 ** 2 * x4
    omega = -500 + delta * x1 + x2

    return omega


def execute(ii, args):
    res = optimize.minimize(optimizerFunction, [args["x1"], args["x2"], args["x3"], args["x4"]])
    log("Result:", res.x)

    save_results(OUTFOLDER, res.x, ii, arg_dict.get("file_data"))
    log("Finished Program: ", os.getpid())


################################################################################
##### Arguments          #######################################################
os_chdir(__file__)  # Local folder
args = load_arguments()

##### Hyper          ##########################################################
ii = args.hyperparam_ii
hyperparams = pd.read_csv("hyperparams.csv")
arg_dict = hyperparams.iloc[ii].to_dict()


##### Session data   ##########################################################
log("load data setssion", arg_dict.get("file_data"))
load_data_session(arg_dict.get("file_data"), method=arg_dict.get("file_data_method"))


log("script", "start", ii)
execute(ii, arg_dict)
log("script ", "terminated", ii)


"""
###########################################################################################
if __name__ == "__main__":
    os_chdir(__file__)
    args = load_arguments()
    ii = args.hyperparam_ii



    ##### Hyper          ##########################################################
    hyperparams = pd.read_csv("hyperparams.csv")
    arg_dict     = hyperparams.iloc[ii].to_dict()


    ##### Session data   ##########################################################
    log("load data setssion")
    load_data_session( arg_dict.get("file_data") , method = arg_dict.get("file_data_method") )


    log("script", "start", ii)
    execute(ii, arg_dict)



    log("script ", "terminated", ii)
"""
