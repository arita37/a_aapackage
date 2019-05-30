# -*- coding: utf-8 -*-
"""
  Daemon monitoring batch
  scans sub-folder in /tasks/
  and execute  /tasks/taskXXXX/main.py in that folder  (if task is not qdone)
                               main.sh
  When task is finished, taskXXX is rename to taskXXX_qdone
_qstart
_qdone  : qdone
_ignore : ignore file
### CLI usage
batch_daemon_launch_cli.py  --task_folder tasks  --log_file zlog/batchdaemong.log  --mode daemon  --waitsec 10  &
### using S3 disk
batch_daemon_launch_cli.py  --task_folder  zs3drive/tasks  --log_file   zlog/batchdaemong.log  --mode daemon  --waitsec 10  
batch_daemon_monitor_cli.py --monitor_log_folder   tasks_out/   --monitor_log_file monitor_log_file.log   --log_file   zlog/batchdaemon_monitor.log    --mode daemon     
########## Details 
batch_daemon_launch_cli.py --param_file zs3drive/config_batch.toml --param_mode test_launch
###############################################################################
cp zs3drive/tasks/ztask_test1_ignore   zs3drive/tasks/task_test1 --recursive
rm zs3drive/tasks/ztask_test1_ignore  --recursive
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from time import sleep

################################################################################
from aapackage import util_log
from aapackage.batch import util_batch, util_cpu

############### logger #########################################################
# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# TASK_FOLDER_DEFAULT = os.getcwd()
TASK_FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__)) + "/ztestasks/"


################################################################################
global logger
logger = logging.basicConfig()


def log(*argv):
    logger.info(",".join([str(x) for x in argv]))


################################################################################
def load_arguments():
    """
       --param_file /zs3drive/config_batch.toml --param_mode test_launch
  """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, "config.toml")

    p = argparse.ArgumentParser()
    p.add_argument("--param_file", default=config_file, help="Params File")
    p.add_argument("--param_mode", default="test", help=" test/ prod /uat")

    p.add_argument("--task_folder", help="path to task folder.")  #     default=TASK_FOLDER_DEFAULT
    p.add_argument("--log_file", help=".")  # default="logfile_batchdaemon.log"
    p.add_argument("--log_file_task", help=".")  #     default="logfile_batchdaemon_task.log",
    p.add_argument("--mode", help="daemon/ .")  #     default="nodaemon",
    p.add_argument("--waitsec", type=int, help="wait sec")  #  default=30,
    p.add_argument(
        "--global_task_file", help="synchronize task"
    )  # default="/home/ubuntu/zs3drive/global_task.json",

    args = p.parse_args()

    ##### Load default file params as dict namespace #########################
    import toml

    class to_namespace(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    try:
        pars = toml.load(args.param_file)  # Load Default params
        # print(pars)
        pars = pars[args.param_mode]  # test / prod
        # print(pars)

        ### Overwrite params by CLI input and merge with toml file
        for key, x in vars(args).items():
            if x is not None:  # only values NOT set by CLI
                pars[key] = x

        print(pars)
        pars = to_namespace(pars)  #  like object/namespace pars.instance
        return pars
    except Exception as e:
        print("Load_argument error", e)
        return args


################################################################################
################################################################################
def get_list_valid_task_folder(folder, script_name="main"):
    if not os.path.isdir(folder):
        return []

    valid_folders = []
    for root, dirs, files in os.walk(folder):
        root_splits = root.split("/")
        for filename in files:
            if (
                filename == "main.sh"
                or filename == "main.py" "_qstart" not in root_splits[-1]
                and "_qdone" not in root_splits[-1]
                and "_ignore" not in root_splits[-1]
            ):
                valid_folders.append(root)

    return valid_folders


def subprocess_launch(foldername, main_file):
    if main_file == "main.py":
        os_python_path = sys.executable
        cmd = [os_python_path, os.path.abspath(foldername + "/" + main_file)]
    else:
        main_file = os.path.abspath(foldername + "/" + main_file)
        os.system("chmod 777 " + main_file)
        cmd = ["bash", main_file]  # main.sh
        # cmd = [ "chmod 777 " + main_file  + " && " + main_file]  # main.sh

    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    return ps.pid


def os_wait_policy(waitsleep=15, cpu_max=95, mem_max=90.0):
    """
      Wait when CPU/Mem  usage is too high 
    """
    from aapackage.batch import util_cpu

    cpu_pct, mem_pct = util_cpu.ps_get_computer_resources_usage()
    log("cpu,mem usage", cpu_pct, mem_pct)
    while cpu_pct > cpu_max or mem_pct > mem_max:
        log("cpu,mem usage", cpu_pct, mem_pct)
        cpu_pct, mem_pct = util_cpu.ps_get_computer_resources_usage()
        time.sleep(waitsleep)


####################################################################################################
####################################################################################################
def isvalid_folder(folder_main, folder, folder_check, global_task_file):
    if os.path.isfile(os.path.join(folder_main, folder)) or folder in folder_check:
        return False
    elif "_qdone" in folder or "_qstart" in folder or "_ignore" in folder:
        global_task_file_save(folder, folder_check, global_task_file)
        return False
    else:
        return True


def global_task_file_save(folder, folder_check, global_task_file):
    folder_check[folder] = time.time()
    json.dump(folder_check, open(global_task_file, mode="w"))
    log("Inserted task", folder, global_task_file)


def main3():
    """ Driver utility for the script.
    global_task_file contains the list of task ALREADY PROCESSED.
    if a task is pick up --> this file is updated Globally.
  
  """
    global logger
    args = load_arguments()
    logger = util_log.logger_setup(
        __name__, log_file=args.log_file, formatter=util_log.FORMATTER_4, isrotate=True
    )

    log("Daemon", "start ", os.getpid())
    folder_main = args.task_folder
    global_task_file = args.global_task_file
    if not os.path.isdir(folder_main):
        return 0

    folder_check = json.load(open(global_task_file, mode="r"))
    while True:
        # log(folder_check)
        log("Daemon new loop", folder_main)
        for folder in os.listdir(folder_main):
            # log(folder)
            if not isvalid_folder(folder_main, folder, folder_check, global_task_file):
                continue

            t0 = time.time()
            folder_check = json.load(open(global_task_file, mode="r"))  # Refresh Global file
            if folder not in folder_check:
                global_task_file_save(
                    folder, folder_check, global_task_file
                )  # Update to prevent 2nd pick up
                log("time to save", time.time() - t0)

                folder = os.path.join(folder_main, folder)
                files = [
                    file for file in os.listdir(folder) if file == "main.sh" or file == "main.py"
                ]
                log(files)
                if files:
                    pid = subprocess_launch(folder, files[0])
                    log("task folder started:", folder, files[0], pid)
                    sleep(20)
                    os_wait_policy(waitsleep=10)

        if args.mode != "daemon":
            log("Daemon", "terminated", os.getpid())
            break

        sleep(args.waitsec)
        os_wait_policy(waitsleep=5)


if __name__ == "__main__":
    main3()


####################################################################################################
####################################################################################################


"""
class global_task(object):
    def __init__(self, task_file):
       self.__dict__.update(adict)
  
    def save(self) :
      pass
      
    def load(self):
      pass
"""


def main():
    """ Driver utility for the script."""
    global logger
    args = load_arguments()
    logger = util_log.logger_setup(
        __name__, log_file=args.log_file, formatter=util_log.FORMATTER_4, isrotate=True
    )

    log("Daemon", "start ", os.getpid())
    while True:
        log("Daemon new loop", args.task_folder)
        folders = get_list_valid_task_folder(args.task_folder)

        if folders:
            log("task folder:", folders)
            pid_list = util_batch.batch_run_infolder(task_folders=folders, log_file=args.log_file)
            log("task folder started:", pid_list)

        if args.mode != "daemon":
            log("Daemon", "terminated", os.getpid())
            break

        sleep(args.waitsec)


def main2():
    """ Driver utility for the script."""
    global logger
    args = load_arguments()
    logger = util_log.logger_setup(
        __name__, log_file=args.log_file, formatter=util_log.FORMATTER_4, isrotate=True
    )

    log("Daemon", "start ", os.getpid())
    folder_main = args.task_folder
    while True:
        log("Daemon new loop", folder_main)
        if not os.path.isdir(folder_main):
            break

        for root, dirs, files in os.walk(folder_main):
            root_splits = root.split("/")
            f = root_splits[-1]
            for filename in files:
                if (
                    filename == "main.sh"
                    or filename == "main.py"
                    and "_qstart" not in f
                    and "_qdone" not in f
                    and "_ignore" not in f
                ):
                    try:
                        #### Issue of collision if 2 instances rename the folder
                        folder_new = root + "_qstart"
                        os.rename(root, folder_new)

                        pid = subprocess_launch(folder_new, filename)
                        log("task folder started:", folder_new, pid)
                    except:
                        pass
                    os_wait_policy(waitsleep=5)

        if args.mode != "daemon":
            log("Daemon", "terminated", os.getpid())
            break

        sleep(args.waitsec)
        os_wait_policy(waitsleep=5)


"""
################### Argument catching ########################################################
print('Start Args')
xprint sys.argv, "\n\n", sys.argv[0], "\n\n", sys.argv[1], "\n\n"
title1= sys.argv[0]
task_folder= title1.split('/')[-2]            #  print 'Task Folder', task_folder
#Task data, str  --> Tuple of arguments
args1=  eval(sys.argv[1])
input_param_file, itask0, itask1= args1[0]
#Other data
args2= args1[1]
print(title1, input_param_file, (itask0, itask1), args2 )
################### Batch Params #############################################################
date0= util.date_now()
isverbose =0
batchname=        task_folder
DIRBATCH=         DIRCWD + '/linux/batch/task/'+ task_folder + '/'
batch_script=     DIRBATCH + '/elvis_pygmo_optim.py'
batch_in_data1=   DIRBATCH + '/aafolio_storage_ref.pkl'
filedata=         DIRBATCH + '/close_15assets_2012_dec216.spydata'
dir_batch_main=   DIRCWD + '/linux/batch/'
batch_out_name=   'batch_' + util.date_nowtime('stamp')   #str(np.random.randint(1000, 99999999))
batch_out=        dir_batch_main + '/output/'+date0 + '/'+ batch_out_name
os.makedirs(batch_out)
batch_out_log=    batch_out + '/output_result.txt'
batch_out_data=   batch_out + '/aafolio_storage_' + date0 + '.pkl'
util.os_print_tofile( '\n\n'+title1, batch_out_log)
################  Output Data table ######################################################
if util.os_file_exist(batch_out_data):
  aux3_cols, aafolio_storage=  util.py_load_obj( batch_out_data, isabsolutpath=1 )
else :
  aux3_cols, aafolio_storage=  util.py_load_obj( batch_in_data1, isabsolutpath=1 )
################## Model Parameters ######################################################
# input_param_file=    DIRBATCH_local+'input_params_all.pkl'
input_params_list=   util.py_load_obj(input_param_file, isabsolutpath=1)
input_params_list=   input_params_list[itask0:itask1,:]    #Reduce the size
############### Loop on each parameters sets #############################################
iibatch=0; krepeat= 1;ii=0
for ii in xrange(itask0-itask0, itask1-itask0):
  iibatch+=1
  params_dict= dict(input_params_list[ii,1])
  globals().update(params_dict)     #Update in the interpreter memory
  util.os_print_tofile('\n ' + task_folder + ' , task_' + str(ii+itask0) + '\n', batch_out_log)
  for kkk in xrange(0, krepeat) :   # Random Start loop  krepeat to have many random start....
      execfile(batch_script)
"""

"""  Manual test
  ii= 2; kkk=1
  params_dict= dict(input_params_list[ii,1])
  globals().update(params_dict)
  util.os_print_tofile('\n task_' + str(ii) +'\n' , batch_out_log)
  for kkk in xrange(0, krepeat) : # Random Start loop
      execfile(batch_script)
"""

#####################################################################################

""" Create Storage file :
np.concatenate((    ))
aafolioref= aafolio_storage[0,:].reshape(1,20)
util.py_save_obj( (aux3_cols, aafolioref) ,  'aafolio_storage_ref' )
aux3_cols, aafolio_storage=  util.py_load_obj( dir_batch_main+  '/batch_20161228_96627894/aafolio_storage_20161227',
                                                   isabsolutpath=1 )
"""

#####################################################################################

"""
for arg in sys.argv:
    print arg
Each command-line argument passed to the program will be in sys.argv, which is just a list. Here you are printing each argument on a separate line.
Example 10.21. The contents of sys.argv
[you@localhost py]$ python argecho.py             1
argecho.py
"""

"""
import argparse
p.add_argument('-i','--input', help='Script File Name', required=False)
p.add_argument('-o','--output',help='Script ID', required=False)
args = p.parse_args()
## show values ##
print ("Input file: %s" % args.input )
print ("Output file: %s" % args.output )
"""
