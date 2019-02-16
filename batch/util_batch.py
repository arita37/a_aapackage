# -*- coding: utf-8 -*-
"""
Confusion comes there 3 levels of Batchs in my initial code

"""

import numpy as np
import os
import shutil
import sys
import random
import psutil

import toml
import subprocess
import argparse
import pandas as pd, numpy as pd
import time
import socket
import logging
import arrow
import util_cpu
################### Argument catching #########################################
global APP_ID, APP_ID2

APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
APP_ID2 = str(os.getpid()) + '_' + str(socket.gethostname())

logging.basicConfig(level=logging.INFO)

def log(s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='',
        s9='', s10=''):
    try:
        prefix = APP_ID + ',' + arrow.utcnow().to('Japan').format(
            "YYYYMMDD_HHmmss,") + ',' 
        s = ','.join([prefix, str(s), str(s1), str(s2), str(s3), str(s4), str(s5),
             str(s6), str(s7), str(s8), str(s9), str(s10)])

        # logging.info(s)
        print(s)

    except Exception as e:
        # logging.info(e)
        print(e)



def checkAdditionalFiles(WorkingDirectory, AdditionalFiles):
    if not AdditionalFiles:
        return []

    AdditionalFiles = [f.strip(" ")
                       for f in AdditionalFiles.split(",")]

    for File in AdditionalFiles:
        ExpectedFilePath = os.path.isfile(os.path.join(WorkingDirectory, File))
        if not ExpectedFilePath:
            print("Additional file <%s> %s not found. Aborting!" % File)
            exit(1)

    return AdditionalFiles


def os_create_folder(WorkingDirectory, folderName):
    folderPath = os.path.join(WorkingDirectory, folderName)
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)



############### Loop on each parameters sets ##################################
def batch_execute_parallel(HyperParametersFile, 
                           subprocess_script, batch_log_file ="batch_logfile.txt",
                           waitseconds = 2):
    """
    task/
          main.py
          subprocess.py
          hyperparams.csv
          ...
    
    
    """
    python_path = sys.executable
        
    HyperParameters  = pd.read_csv( HyperParametersFile)
    OptimizerName    = os.path.basename(subprocess_script)
    WorkingDirectory = subprocess_script

    # logging.basicConfig(level=logging.INFO)
    #batch_log_file = os.path.join(WorkingDirectory, "batch_logs/batch_%s.txt" % batch_label) 
    ChildProcesses = []

    for ii in range(HyperParameters.shape[0]):
       # Extract parameters for single run from batch_parameters data.
       # params_dict = HyperParameters.iloc[ii].to_dict()
       log(batch_label, "Executing index", ii, WorkingDirectory, "\n\n")

       proc = subprocess.Popen([ python_path, subprocess_script, str(ii) ],
                                 stdout=subprocess.PIPE)
       # ChildProcesses.append(proc)

       # wait if computer resources are scarce.
       cpu, mem = 100, 100
       while cpu > 90 and mem > 90:
          cpu, mem = util_cpu.ps_get_computer_resources_usage()
          time.sleep(waitseconds)




##########################################################################################
##########################################################################################
def batch_generate_hyperparameters(hyper_dict,file_hyper) :
  """
     {  "layer" : {"min": 10  , "max": 200 , "type": int,    "nmax": 10, "method": "random"  },
        "layer" : {"min": 10  , "max": 200 , "type": float,  "nmax": 10  }, "method": "linear"
     }

    size : key1 x ke2 x key2

  """
  for key  in hyper_dict:
       vv = np.arange( hyper_dict["key"]["min"]  ,   hyper_dict["key"]["max"] )
       df = df.extend(  len(vv) )

  df.to_csv( file_hyper )
   

def ps_wait_for_completion(subprocess_list):
    for pid in subprocess_list:
        while True:
            try:
                pr = psutil.Process(pid)
                try:
                    pr_status = pr.status()
                except TypeError:  # psutil < 2.0
                    pr_status = pr.status
                except psutil.NoSuchProcess:  # pragma: no cover
                    break
                # Check if process status indicates we should exit
                if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    break
            except:
                break
            time.sleep(1)


def get_python_executable():
    return str(sys.executable)


def sp_execute_script(execute=get_python_executable(), script="", stdout=subprocess.PIPE, stderr=subprocess.STDOUT):
    ps = subprocess.Popen([execute, script], stdout=stdout, stderr=stderr, shell=False)
    return ps


def os_rename_folder(old_directory , new_directory):
    os.rename(old_directory, new_directory)


def os_folder_create(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

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














