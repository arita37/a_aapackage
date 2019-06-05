# -*- coding: utf-8 -*-
"""
Run task
python main.py  --hyperparam      --subprocess_script sub


"""

import argparse
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from functools import partial

import arrow
import numpy as np
import pandas as pd
import psutil
import toml

################### Generic ###############################################################
from aapackage import util_log
from aapackage.batch import util_batch
from utils import OUTFOLDER, os_folder_create, os_getparent

###########################################################################################
CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HYPERPARAMS = os.path.join(CUR_FOLDER, "hyperparams.csv")
DEFAULT_SUBPROCESS_SCRIPT = os.path.join(CUR_FOLDER, "subprocess_pygmo.py")


##### Logs     ############################################################################
os_folder_create(OUTFOLDER)
APP_ID = util_log.create_appid(__file__)
LOG_FILE = os.path.join(OUTFOLDER, "log_main.log")


def log(s="", s1=""):
    util_log.printlog(s=s, s1=s1, app_id=APP_ID, logfile=LOG_FILE)


##### Args     ############################################################################
def load_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hp",
        "--hyperparam_file",
        default=DEFAULT_HYPERPARAMS,
        type=str,
        help="Select the path for a .csv.",
    )

    parser.add_argument(
        "-s",
        "--subprocess_script",
        default=DEFAULT_SUBPROCESS_SCRIPT,
        type=str,
        help="Name of the optimizer script.",
    )

    parser.add_argument("-f", "--task_folder", default=CUR_FOLDER, type=str, help="W")
    parser.add_argument("-l", "--log_file", default=LOG_FILE, type=str, help="W")
    parser.add_argument("-o", "--out_folder", default=OUTFOLDER, type=str, help="W")

    options = parser.parse_args()
    return options


############################################################################################
def os_folder_rename(task_folder):
    # After termination of script
    k = task_folder.rfind("qstart")
    new_name = task_folder[:k] + "qdone"
    os.rename(task_folder, new_name)


############################################################################################
if __name__ == "__main__":
    log("main.py", "start")
    args = load_arguments()

    util_batch.batch_parallel_subprocess(
        args.hyperparam_file, args.subprocess_script, args.log_file
    )

    os_folder_rename(args.task_folder)
    log("main.py", "finish")


"""

def execute_script(hyperparam, subprocess_script, file_logs, row_number):
    cmd_list = [PYTHON_COMMAND, subprocess_script, "--hyperparam_ii=%d" % row_number]
    ps = subprocess.Popen( cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    log("Subprocess started by execute_script: %s" % str(ps.pid))
    return ps.pid



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

batch_out_logs = batch_out + '/output_result.txt'
batch_out_data = batch_out + '/aafolio_storage_' + date0 + '.pkl'
util.os_print_tofile('\n\n'+title1, batch_out_logs)
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
