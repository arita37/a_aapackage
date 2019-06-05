# -*- coding: utf-8 -*-
"""
hyperparams
  file_log
  file_data
  file_data_method



"""
import os
import sys

import arrow
import pandas as pd
import toml

import pygmo as pg
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


###############################################################################
##### Scripts  ################################################################
log("Start Script", __file__)


# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
prob = pg.problem(pg.schwefel(5))

# 2 - Instantiate a pagmo algorithm
algo = pg.algorithm(pg.sade(gen=10))

# 3 - Instantiate an archipelago with 16 islands having each 20 individuals
archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=10)

# 4 - Run the evolution in parallel on the 16 separate islands 10 times.
archi.evolve(5)

# 5 - Wait for the evolutions to be finished
archi.wait()

# 6 - Print the fitness of the best solution in each island
res = [isl.get_population().champion_f for isl in archi]
print(res)


###############################################################################
save_results(OUTFOLDER, res.x, ii, arg_dict.get("file_data"))
log("Finished Program: ", os.getpid())


###############################################################################
##### End Script Rnme Folder ##################################################
#### Rename folder ################
