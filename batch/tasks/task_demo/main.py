# -*- coding: utf-8 -*-
"""
Run task

python main.py  --hyperparam      --subprocess_script sub


"""
import numpy as np , pandas as pd
import os, sys
import shutil
import random
import psutil

import toml
import subprocess
import argparse
import time

  
from aapackage._batch.util_batch import *


###############################################################################
################### Argument catching #########################################
def os_getparent(dir0):
    return os.path.abspath(os.path.join(dir0, os.pardir))


def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", "--hyperparam", default="hyperparams.csv", type=str,
                      help="Select the path for a .csv containing batch optimization parameters.\n One row per optimization.")

    parser.add_argument("-s", "--subprocess_script",  default="subprocess_test.py", type=str,
                      help="Name of the optimizer script. Should be located at WorkingDirectory")

    parser.add_argument("-f", "--file_log", default="log_main.txt", type=str,
                      help="Wether optimization will run locally or on a dedicated AWS instance.")

    options = parser.parse_args()
    return options



if __name__ != '__main__' :
    args = load_arguments()

    WorkingDirectory = os.path.dirname(os.path.abspath(__file__))
    printlog("start task main", __file__)
    
    batch_execute_parallel( args.hyperparam, 
                            WorkingDirectory,
                            args.subprocess_script, 
                            args.file_log)












"""
date0 = util.date_now()
isverbose = 0
batchname = task_folder
DIRBATCH = DIRCWD + '/linux/batch/task/' + task_folder + '/'

batch_script = DIRBATCH + '/elvis_pygmo_optim.py'
batch_in_data1 = DIRBATCH + '/aafolio_storage_ref.pkl'
filedata = DIRBATCH + '/close_15assets_2012_dec216.spydata'


dir_batch_main = DIRCWD + '/linux/batch/'
# str(np.random.randint(1000, 99999999))
batch_out_name = 'batch_' + util.date_nowtime('stamp')
batch_out = dir_batch_main + '/output/'+date0 + '/' + batch_out_name
os.makedirs(batch_out)

batch_out_log = batch_out + '/output_result.txt'
batch_out_data = batch_out + '/aafolio_storage_' + date0 + '.pkl'
util.os_print_tofile('\n\n'+title1, batch_out_log)
"""



"""
if util.os_file_exist(batch_out_data):
    aux3_cols, aafolio_storage = util.py_load_obj(
        batch_out_data, isabsolutpath=1)
else:
    aux3_cols, aafolio_storage = util.py_load_obj(
        batch_in_data1, isabsolutpath=1)


"""



"""
        No Need t 
        # build path & create task directory;
        TaskDirectoryName = "task_%s_%i" % (batch_label, ii)
        TaskDirectory = os.path.join(WorkingDirectory, TaskDirectoryName)

        os.mkdir(TaskDirectory)

        # Copy optimization script to Task Directory;
        shutil.copy(os.path.join(WorkingDirectory, OptimizerName),
                    os.path.join(TaskDirectory, OptimizerName))

        # Save Parameters inside Task Directory;
        with open(os.path.join(TaskDirectory, "parameters.toml"), "w") as task_parameters:
            toml.dump(params_dict, task_parameters)

        # Copy additional files to Task Directory;
        if AdditionalFilesArg:
            AdditionalFiles = checkAdditionalFiles(WorkingDirectory, AdditionalFilesArg)
            for File in AdditionalFiles:
                shutil.copy(os.path.join(WorkingDirectory, File),
                            os.path.join(TaskDirectory, File))
"""














