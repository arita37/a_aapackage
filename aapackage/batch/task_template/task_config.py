# -*- coding: utf-8 -*-
"""
File containing task configuration and utilities to save data on S3


Folders :
   task_name :    mytask
   taskout_local   :  /home/ubuntu/tasks/mytask/
   taskout_s3_root :  /home/ubuntu/zs3tasks/tasks/
  


"""
import os
import random
import sys
from time import sleep

HOME_DIR = os.environ["HOME"] if "HOME" in os.environ else "/home/ubuntu"


######### Variables ################################################################################
task_cpu_required = 2
taskout_s3_root = HOME_DIR + "/zs3drive/tasks_out/"  # S3 Global Drive
taskout_local_root = HOME_DIR + "/tasks_out/"  # Instance Local Drive


######### Folder ###################################################################################
# print( __file__ )
task_name = __file__.split("/")[-2]  # Task name IS folder name
taskout_local = taskout_local_root + task_name + "/"  # Local Instance Drive
taskout_s3 = taskout_s3_root + task_name + "/"  # Global Drive


if not os.path.exists(taskout_local):
    msg = os.system("mkdir " + taskout_local)


# if os.path.exists( taskout_local ) :
#   print("Task out Local Folder already exist")
# print( local_taskout, s3_taskout)


######### Utils ####################################################################################
def os_copy_local_to_s3(taskout_local, taskout_s3_root):
    """
    Copy to Local DRIVE to Global File System S3  (ie Spot Instance)
  """

    task_name = taskout_local.split("/")[-1]
    if not os.path.exists(taskout_s3_root):
        os.system("mkdir " + taskout_s3_root)

    if os.path.exists(taskout_s3_root + "/" + task_name):
        print("Task out s3 Folder already exist, Overwriting", taskout_s3_root + "/" + task_name)

    cmd = " cp -r {a}  {b}".format(a=taskout_local, b=taskout_s3_root)
    msg = os.system(cmd)
    print("Copy success", msg)


def os_rename_taskfolder(task_name, taskout_s3_root, suffix="_qdone"):
    """
    Task done, renamed
  
  """
    task_folder = taskout_s3_root + "/" + task_name
    task_folder_new = task_folder + "_qdone"

    if os.path.exists(task_folder_new):
        task_folder_new = task_folder + "_qdone" + "_" + str(random.randint(100, 1000))

    cmds = "cp  {a}  {b} --recursive  ".format(a=task_folder, b=task_folder_new)
    cmds += " && rm {a}   --recursive  ".format(a=task_folder)
    print(cmds)
    msg = os.system(cmds)
    print(msg)
    # msg = os.system(" ls " + task_folder_new)
    # print(msg)


####################################################################################################
######### CLi ######################################################################################
def load_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="batchdaemon_autoscale.log", help=".")
    parser.add_argument("--do", default="ncpu_required/ram", help="daemon/ .")
    options = parser.parse_args()
    return options


# print("test")


if __name__ == "__main__":
    args = load_arguments()

    if args.do == "ncpu_required":
        print(task_cpu_required)
