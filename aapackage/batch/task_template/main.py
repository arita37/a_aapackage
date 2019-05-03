# -*- coding: utf-8 -*-
"""
python zs3drive/tasks/t_github_test02/main.py


Folders :
  taskout_local  : /home/ubuntu/


"""
import os, sys
from time import sleep

####################################################################################################
from util_task import task_name, taskout_local, taskout_s3_root, os_copy_local_to_s3     




####################################################################################################
out_file =  taskout_local + "/myoutput.txt"


with open(out_file, mode="a") as f :
  for i in range(0, 15) :
    f.write(  "{i} \n".format(i=i)   )
    # sleep(1)


















####################################################################################################
######## Export ####################################################################################
os_copy_local_to_s3( taskout_local, taskout_s3_root ) 


#### Rename taks folder
# os_rename_task(task_name, taskout_s3_root )
  

####################################################################################################







