# -*- coding: utf-8 -*-
"""
Run task

python main.py  --hyperparam      --subprocess_script sub


"""
import numpy as np, pandas as pd
import os, sys
import shutil
import random
import psutil

import toml
import subprocess
import argparse
import time

# from aapackage._batch.util_batch import *
import logging
import psutil

logging.basicConfig(level=logging.INFO)
STDOUT_FORMAT = "Finished Program: %s"
working_directory = os.path.dirname(os.path.abspath(__file__))

DEFAULT_HYPERPARAMS = os.path.join(working_directory, "hyperparams.csv")
DEFAULT_SUBPROCESS_SCRIPT = os.path.join(working_directory, "subprocess_optim.py")


###############################################################################
################### Argument catching #########################################
def os_getparent(dir0):
    return os.path.abspath(os.path.join(dir0, os.pardir))


def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyperparam", default=DEFAULT_HYPERPARAMS, type=str,
                        help="Select the path for a .csv containing batch optimization parameters.One row per "
                             "optimization.")

    parser.add_argument("-s", "--subprocess_script", default=DEFAULT_SUBPROCESS_SCRIPT, type=str,
                        help="Name of the optimizer script. Should be located at WorkingDirectory")

    parser.add_argument("-f", "--file_log", default="log_main.txt", type=str,
                        help="Wether optimization will run locally or on a dedicated AWS instance.")

    options = parser.parse_args()
    return options


def execute_script(hyperparam, subprocess_script, file_log, row_number):
    ps = subprocess.Popen(["python %s %s" % (subprocess_script, str(row_number))], shell=True, stdout=subprocess.PIPE)
    print("Subprocess started by execute_script: %s" % str(ps.pid))
    return ps


# def kill_process_after_completion(subprocess_list):
#     while len(subprocess_list) > 0:
#         for sp in subprocess_list:
#             while True:
#                 line = sp.stdout.readline()
#                 if line != '' and line.rstrip() == STDOUT_FORMAT % str(sp.pid):
#                     print("Subprocess Killed with pid: %s" % str(sp.pid))
#                     sp.kill()
#                     subprocess_list.remove(sp)
#                     break
#                 else:
#                     break
#         time.sleep(1)


def rename_directory(working_directory):
    k = working_directory.rfind("qstart")
    new_name = working_directory[:k] + "qdone"
    os.rename(working_directory, new_name)


if __name__ == '__main__':
    args = load_arguments()

    subprocess_script = args.subprocess_script
    file_log = args.file_log
    hyperparam = args.hyperparam

    print ("start task main", __file__)

    hyper_parameter_set = pd.read_csv(hyperparam)
    rows_length = len(hyper_parameter_set.index)
    subprocess_list = []
    for each_row in range(0, rows_length):
        subprocess_list.append(execute_script(hyperparam, subprocess_script, file_log, each_row))
        time.sleep(5)

    rename_directory(working_directory=working_directory)
    print("Finished Program: %s" % str(os.getpid()))

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
