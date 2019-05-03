# -*- coding: utf-8 -*-
"""
File containing task configuration and utilities to save data on S3


Folders :
  task_name :    mytask
  taskout_local   :  /home/ubuntu/tasks/mytask/
  taskout_s3_root :  /home/ubuntu/zs3tasks/tasks/
  


"""
import os, sys
from time import sleep
import random
####################################################################################################


task_cpu = 2
HOME_DIR = os.environ['HOME']  if 'HOME' in os.environ else '/home/ubuntu'
taskout_s3_root    =  HOME_DIR + "/zs3drive/tasks_out/" 
taskout_local_root =  HOME_DIR +  "/tasks_out/" 




######### Folder ###################################################################################
# print( __file__ )
task_name     =  __file__.split("/")[-2]
taskout_local = taskout_local_root + task_name  + "/" 
taskout_s3    = taskout_s3_root    + task_name + "/"

if os.path.exists( taskout_local ) :
   print("Task out Local Folder already exist")
else :
   os.system("mkdir " + taskout_local )
   
####################################################################################################

# print( local_taskout, s3_taskout)


def os_copy_local_to_s3( taskout_local, taskout_s3_root ) :
  """
    Copy to Local DRIVE to Global File System S3  (ie Spot Instance)
  """
  
  task_name =  taskout_local.split("/")[-1]
  if not os.path.exists(  taskout_s3_root   ) :
    os.system("mkdir " + taskout_s3_root   )

  if os.path.exists(  taskout_s3_root + "/" + task_name   ) :
    print("Task out s3 Folder already exist, Overwriting", taskout_s3_root + "/" + task_name   )


  cmd =  " cp -r {a}  {b}".format(a=taskout_local, b=taskout_s3_root  )
  msg = os.system(cmd)
  print("Copy success", msg)
 
 


def os_rename_taskfolder(task_name, taskout_s3_root , suffix="_qdone"):
  """
    Task done, renamed
  
  """
  task_folder     =  taskout_s3_root + "/" + task_name
  task_folder_new =  task_folder + "_qdone"
    
  if os.path.exists(task_folder_new) :
    task_folder_new =  task_folder +  "_qdone" + "_" +str( random.randint(100,1000)) 
  
  cmds  = "cp  {a}  {b} --recursive  ".format(a=task_folder, b=task_folder_new)
  cmds += " && rm {a}   --recursive  ".format(a=task_folder)
  print(cmds)
  msg = os.system(cmds)
  print(msg)
  # msg = os.system(" ls " + task_folder_new)
  # print(msg)
    


"""





"""
