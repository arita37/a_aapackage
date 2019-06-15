# -*- coding: utf-8 -*-
"""
Batch utils
    main.sh
    main.py



"""
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from aapackage import util_log
#######################################################################

################### Variables #########################################


######### Logging ####################################################
global logger
LOG_FILE = "zlog/" + util_log.create_logfilename(__file__)
APP_ID = util_log.create_appid(__file__)

# noinspection PyRedeclaration
logger = util_log.logger_setup(__name__, log_file=None, formatter=util_log.FORMATTER_4)


def log(*argv):
    logger.info(",".join([str(x) for x in argv]))


# log("Ok, test_log")
#####################################################################


######################################################################
def os_python_path():
    return str(sys.executable)


def os_folder_rename(old_folder, new_folder):
    try:
        if os.path.isdir(new_folder):
            new_folder = new_folder + str(random.randint(100, 999))
        os.rename(old_folder, new_folder)
        return new_folder
    except Exception as e:
        return old_folder


def os_folder_create(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def os_cmd_generate(task_folder, os_python_path=None):
    """
   params.toml check
   otherwise default config is/main.py with current interpreter.
     main.sh 
     main.py
     
  """
    main_file = os.path.join(task_folder, "main.sh")
    if os.path.isfile(main_file):
        cmd = [main_file]
        return cmd

    main_file = os.path.join(task_folder, "main.py")
    if os.path.isfile(main_file):
        os_python_path = sys.executable if os_python_path is None else os_python_path
        cmd = [os_python_path, main_file]
        return cmd


def os_wait_policy(waitsleep=15, cpu_max=95, mem_max=90.0):
    """
      Wait when CPU/Mem  usage is too high 
    
    """
    from aapackage.batch import util_cpu

    cpu_pct, mem_pct = util_cpu.ps_get_computer_resources_usage()
    while cpu_pct > cpu_max or mem_pct > mem_max:
        log("cpu,mem usage", cpu_pct, mem_pct)
        cpu_pct, mem_pct = util_cpu.ps_get_computer_resources_usage()
        time.sleep(waitsleep)


def batch_run_infolder(
    task_folders,
    suffix="_qstart",
    main_file_run="main.py",
    waitime=7,
    os_python_path=None,
    log_file=None,
):
    sub_process_list = []
    global logger

    if ".py" in main_file_run:
        ispython = 1

    if os_python_path is None:
        os_python_path = sys.executable

    if log_file is not None:
        logger = util_log.logger_setup(__name__, log_file=log_file, formatter=util_log.FORMATTER_4)

    for folder_i in task_folders:
        foldername = folder_i + suffix
        foldername = os_folder_rename(old_folder=folder_i, new_folder=foldername)

        # main_file = os.path.join(foldername,  main_file_run )
        # cmd = [os_python_path, main_file]  if ispython else [ main_file]
        cmd = os_cmd_generate(foldername, os_python_path)

        os_wait_policy(waitsleep=15)
        ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        sub_process_list.append(ps.pid)

        log("Sub-process, ", ps.pid, cmd)
        time.sleep(waitime)

    return sub_process_list


def batch_parallel_subprocess(hyperparam_file, subprocess_script, os_python_path=None, waitime=5):
    """
    task/
          main.py
          subprocess.py
          hyperparams.csv
          ...
          :type hyperparam_file: str

    """
    hyper_parameter = pd.read_csv(hyperparam_file)
    PYTHON_PATH = sys.executable if os_python_path is None else os_python_path
    ispython = 1 if ".py" in subprocess_script else 0

    # Start process launch
    subprocess_list = []
    for ii in range(0, len(hyper_parameter)):
        if ispython:
            cmd = [PYTHON_PATH, subprocess_script, "--hyperparam_ii=%d" % ii]
        else:
            cmd = [subprocess_script, ii]

        ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        log("Subprocess started ", ps.pid)
        subprocess_list.append(ps.pid)
        time.sleep(waitime)
        # util_cpu.ps_wait_ressourcefree(cpu_max=90, mem_max=90, waitsec=15)

    # util_cpu.ps_wait_process_completion(subprocess_list)


def batch_generate_hyperparameters(hyperparam_dict, outfile_hyperparam):
    """
     {  "param1" : {"min": 10  , "max": 200 , "type": int,    "nmax": 10, "method": "random"  },
        "param2" : {"min": 10  , "max": 50 , "type": float,  "nmax": 10  }, "method": "linear"
     }

    df
      nbcols    :  range for param1  X  range for param2  X range for param3
      nbcolumns :  len(hyper_dict)

   https://stackoverflow.com/questions/42795832/faster-way-to-extend-values-of-a-pandas-dataframe

   Dict preserving the order !!!

    Equivalent of grid search
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


  """
    # df not defined , DataFrame has no "extend" method
    ##init columns

    df = pd.DataFrame({k for k in hyperparam_dict.keys()})

    for key, d in hyperparam_dict.items():
        df[key] = 0

        vv = np.arange(d["min"], d["max"], d["nmax"])

        # df = df.extend(  len(vv) )

    # df.to_csv( file_hyper )
    return 1






if __name__ == "__main__":
    ################## Initialization #########################################
    log(" Log check")

    ############## RUN Monitor ################################################
    # monitor()


"""
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
