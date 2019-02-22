# -*- coding: utf-8 -*-
"""
batch utils


"""
import os
import shutil
import sys
import random
import subprocess
import argparse
import time
import logging


import psutil
import toml
import pandas as pd, numpy as pd
import arrow


#######################################################################
from aapackage.batch import util_cpu
from aapackage import util_log



################### Variables #########################################





######### Logging ####################################################
APP_ID  = util_log.create_appid( __file__ )
def log(m):
    util_log.printlog(s1=m, app_id=APP_ID)




######################################################################
def ps_wait_completion(subprocess_list):
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


def os_python_path():
    return str(sys.executable)


def os_folder_rename(old_folder, new_folder):
    try :
       os.rename(old_folder, new_folder)
    except Exception as e :
       os.rename(old_folder, new_folder + str(random.randint(100, 999)) )

def os_folder_create(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def batch_run_infolder(valid_folders, suffix="_qstart", main_file_run="main.py", waitsleep=5 ):
    sub_process_list = []
    for folder_i in valid_folders:
        foldername = folder_i + suffix
        os_folder_rename(old_folder=folder_i, new_folder=foldername)

        main_file = os.path.join(foldername,  main_file_run )
        log("running file: %s" % main_file)
        cmd = [os_python_path(), main_file]
        ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        sub_process_list.append(ps.pid)

        log("running process: %s" % str(ps.pid))
        time.sleep( waitsleep )

    return sub_process_list


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
  pass


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
    WorkingFolder = subprocess_script

    # logging.basicConfig(level=logging.INFO)
    #batch_log_file = os.path.join(WorkingFolder, "batch_logs/batch_%s.txt" % batch_label)
    ChildProcesses = []

    for ii in range(HyperParameters.shape[0]):
       # Extract parameters for single run from batch_parameters data.
       # params_dict = HyperParameters.iloc[ii].to_dict()
       log(batch_label, "Executing index", ii, WorkingFolder, "\n\n")

       proc = subprocess.Popen([ python_path, subprocess_script, str(ii) ],
                                 stdout=subprocess.PIPE)
       # ChildProcesses.append(proc)

       # wait if computer resources are scarce.
       cpu, mem = 100, 100
       while cpu > 90 and mem > 90:
          cpu, mem = util_cpu.ps_get_computer_resources_usage()
          time.sleep(waitseconds)




'''
def checkAdditionalFiles(WorkingFolder, AdditionalFiles):
    if not AdditionalFiles:
        return []

    AdditionalFiles = [f.strip(" ")
                       for f in AdditionalFiles.split(",")]

    for File in AdditionalFiles:
        ExpectedFilePath = os.path.isfile(os.path.join(WorkingFolder, File))
        if not ExpectedFilePath:
            print("Additional file <%s> %s not found. Aborting!" % File)
            exit(1)

    return AdditionalFiles


def os_create_folder(WorkingFolder, folderName):
    folderPath = os.path.join(WorkingFolder, folderName)
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)





'''







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
        TaskFolderName = "task_%s_%i" % (batch_label, ii)
        TaskFolder = os.path.join(WorkingFolder, TaskFolderName)

        os.mkdir(TaskFolder)

        # Copy optimization script to Task Folder;
        shutil.copy(os.path.join(WorkingFolder, OptimizerName),
                    os.path.join(TaskFolder, OptimizerName))

        # Save Parameters inside Task Folder;
        with open(os.path.join(TaskFolder, "parameters.toml"), "w") as task_parameters:
            toml.dump(params_dict, task_parameters)

        # Copy additional files to Task Folder;
        if AdditionalFilesArg:
            AdditionalFiles = checkAdditionalFiles(WorkingFolder, AdditionalFilesArg)
            for File in AdditionalFiles:
                shutil.copy(os.path.join(WorkingFolder, File),
                            os.path.join(TaskFolder, File))
"""














