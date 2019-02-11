# -*- coding: utf-8 -*-
"""
Run task

python main.py  --hyperparam      --subprocess_script sub


"""

import os, sys, socket
import shutil
import random
from functools import partial 

import subprocess
import argparse
import time

# from aapackage._batch.util_batch import *
import logging
import toml
import psutil
import arrow
import numpy as np, pandas as pd


###########################################################################################
################### Generic ###############################################################
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../..")
from utils import logs, os_getparent, ps_wait_process_complete



###########################################################################################
###########################################################################################
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HYPERPARAMS       = os.path.join(WORKING_DIRECTORY, "hyperparams.csv")
DEFAULT_SUBPROCESS_SCRIPT = os.path.join(WORKING_DIRECTORY, "subprocess_optim.py")
PYTHON_COMMAND = sys.executable



##### Logss     ###########################################################################
logging.basicConfig( level=logging.INFO )
log = partial(logs, APP_ID= os.path.abspath(__file__) + ',' + str(os.getpid()) + ',' + str(socket.gethostname()))






def batch_parallel_subprocess(hyperparam_file,  subprocess_script, file_logs ) :
    hyper_parameter = pd.read_csv(hyperparam_file)
    
    # Start process launch
    subprocess_list = []
    for ii in range(0, len(hyper_parameter) ):
        pid = execute_script( hyperparam_file, subprocess_script, file_logs, ii)
        
        subprocess_list.append(pid)
        time.sleep(5)

    ps_wait_process_complete(subprocess_list)
    os_folder_rename(working_directory=WORKING_DIRECTORY)

    log("Finished Program:" , __file__, str(os.getpid()))



def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyperparam_file", default=DEFAULT_HYPERPARAMS, type=str,
                        help="Select the path for a .csv containing batch optimization parameters.One row per "
                             "optimization.")

    parser.add_argument("-s", "--subprocess_script", default=DEFAULT_SUBPROCESS_SCRIPT, type=str,
                        help="Name of the optimizer script. Should be located at WorkingDirectory")

    parser.add_argument("-f", "--file_logs", default="logs_main.txt", type=str,   help="W")

    options = parser.parse_args()
    return options



def execute_script(hyperparam, subprocess_script, file_logs, row_number):
    cmd_list = [PYTHON_COMMAND, subprocess_script, str(row_number)]
    ps = subprocess.Popen( cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    log("Subprocess started by execute_script: %s" % str(ps.pid))
    return ps.pid



def os_folder_rename(working_directory):
    # After termination of script
    k = working_directory.rfind("qstart")
    new_name = working_directory[:k] + "qdone"
    os.rename(working_directory, new_name)





if __name__ == '__main__':
    log("start task main", __file__)
    args = load_arguments()    
    
    batch_parallel_subprocess(args.hyperparam_file,  args.subprocess_script, args.file_logs) 

    """
    hyper_parameter = pd.read_csv(args.hyperparam)
    

    # Start process launch
    subprocess_list = []
    for ii in range(0, len(hyper_parameter) ):
        pid = execute_script( args.hyperparam, args.subprocess_script, args.file_logs, ii)
        
        subprocess_list.append(pid)
        time.sleep(5)

    ps_wait_process_complete(subprocess_list)
    os_folder_rename(working_directory=WORKING_DIRECTORY)

    log("Finished Program:" , __file__, str(os.getpid()))
    """ 















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
