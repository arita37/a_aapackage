# -*- coding: utf-8 -*-
"""
Confusion comes there 3 levels of Batchs in my initial code

meta_batch_task.py :
   ...task_folder/task1/mybatch_optim.py  + hyperparamoptim.csv +  myscript.py ...
   ...task_folder/task2/mybatch_optim.py  + hyperparamoptim.csv + .myscript.py ...


meta_batch_task.py :
   for all "task*" folders in task_working_dir  :
       run subprocess taskXXX/mybatch_optim.py



mybatch_optim.py :
  for all rows ii of hyperparams.csv
     run subprocess  myscript.py  ii
     check CPU suage with psutil.   CPU usage < 90 %   mem usage < 90%


### not needed from you
aws_batch_script.py :
   for all tasks folder in LOCAL PC :
       transfer by SFTP to REMOTE Task folder (by zip)

   subprocess  meta=batch_task.py on REMOTE PC.    


##### Warning :
  Please check my initial code to make sure most of functionnalities are here in your code


####################################################################
batch_sequencer.py :
  Launch 1 subprcoess per hyperparam row.


"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import toml

################### Argument catching #########################################
print("Start Args")
"""
print sys.argv, "\n\n", sys.argv[0], "\n\n", sys.argv[1], "\n\n"
"""


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparam",
        dest="HyperParametersFile",
        help="Select the path for a .csv containing batch optimization parameters.\n One row per optimization.",
    )

    parser.add_argument(
        "--directory",
        dest="WorkingDirectory",
        default=".",
        help="Absolute or relative path to the working directory.",
    )

    parser.add_argument(
        "--optimizer",
        dest="OptimizerName",
        default="optimizer.py",
        help="Name of the optimizer script. Should be located at WorkingDirectory",
    )

    parser.add_argument(
        "--file",
        dest="AdditionalFiles",
        help="A file or comma-separated list of files to be provided for the optimizer function.",
    )

    parser.add_argument(
        "--aws",
        dest="ExecuteAws",
        action="store_true",
        help="Wether optimization will run locally or on a dedicated AWS instance.",
    )

    parser.add_argument(
        "--file_log",
        dest="ExecuteAws",
        action="store_true",
        help="Wether optimization will run locally or on a dedicated AWS instance.",
    )

    options = parser.parse_args()
    return options


################### Batch Params ##############################################
def checkAdditionalFiles(WorkingDirectory, AdditionalFiles):
    if not AdditionalFiles:
        return []

    AdditionalFiles = [f.strip(" ") for f in AdditionalFiles.split(",")]

    for File in AdditionalFiles:
        ExpectedFilePath = os.path.isfile(os.path.join(WorkingDirectory, File))
        if not ExpectedFilePath:
            print("Additional file <%s> %s not found. Aborting!" % File)
            exit(1)

    return AdditionalFiles


################  Output Data table ###########################################
def log(f, m):
    with open(f, "a") as _log:
        _log.write(m)


def get_computer_resources_usage():
    cpu_used_percent = psutil.cpu_percent()

    mem_info = dict(psutil.virtual_memory()._asdict())
    mem_used_percent = 100 - mem_info["available"] / mem_info["total"]

    return cpu_used_percent, mem_used_percent


def createFolder(WorkingDirectory, folderName):
    folderPath = os.path.join(WorkingDirectory, folderName)
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)


############### Loop on each parameters sets ##################################
def batch_execute_parallel(
    HyperParametersFile, WorkingDirectory, ScriptPath, batch_log_file="batch_logfile.txt"
):

    waitseconds = 2
    python_path = sys.executable

    HyperParameters = pd.read_csv(os.path.join(WorkingDirectory, HyperParametersFile))
    OptimizerName = os.path.basename(ScriptPath)

    batch_label = "%s_%.3i" % (OptimizerName, random.randint(0, 10e5))
    batch_log_file = os.path.join(WorkingDirectory, "batch_logs/batch_%s.txt" % batch_label)
    ChildProcesses = []

    for ii in range(HyperParameters.shape[0]):
        # Extract parameters for single run from batch_parameters data.
        params_dict = HyperParameters.iloc[ii].to_dict()

        log(batch_log_file, "Executing index %i at %s." % (ii, WorkingDirectory))
        log(batch_log_file, "\n\n")

        proc = subprocess.Popen([python_path, ScriptPath, str(ii)], stdout=subprocess.PIPE)

        # ChildProcesses.append(proc)

        # wait if computer resources are scarce.
        cpu, mem = 100, 100
        while cpu > 90 and mem > 90:
            cpu, mem = get_computer_resources_usage()
            time.sleep(waitseconds)


if __name__ != "__main__":
    args = load_arguments()
    # init folders
    createFolder(WorkingDirectory, "batch_logs")
    createFolder(WorkingDirectory, "batch_results")


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
