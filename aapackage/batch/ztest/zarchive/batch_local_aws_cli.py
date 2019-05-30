# -*- coding: utf-8 -*-
"""
Many tasks :

Launch AWS server
Send task folder from LOCAL to remote AWS
Retrieve results from AWS folder to local.
Filter out already retrieve results.
Start on AWS the batch_daemon_launchi_cli.py



"""
import argparse
import logging
# -*- coding: utf-8 -*-
import os
import sys
from time import sleep

import numpy as np

import util
################################################################################
from aapackage import util_aws, util_log
from aapackage.batch import util_batch

############### Variable definition ############################################
FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__))



######### Logging ##############################################################
logger = logging.basicConfig()
def log(*argv):
    logger.info( ",".join( [  str(x) for x in argv  ]  ))

# log("Ok, test_log")
################################################################################





################################################################################
def get_list_valid_task_folder(folder, script_name="main.py"):
    if not os.path.isdir(folder):
        return []
    valid_folders = []
    for root, dirs, files in os.walk(folder):
        root_splits = root.split("/")
        for filename in files:
            if  "_qstart" not in root_splits[-1] and  "_qdone" not in root_splits[-1] \
                and filename == script_name and "_ignore" not in root_splits[-1]  :
                valid_folders.append(root)

    return valid_folders



def launch_ec2_(instance_type):
    """
     AWS CLI on windows

     aws ec2 start-instances  --region us-west-2   --instance-ids i-0b197e983c0647053
aws ec2 describe-instances --region us-west-2   --instance-ids i-0b197e983c0647053
pause
ping 127.0.0.1 -n 10 > nul




    """





def task_transfer_to_ec2(fromfolder, tofolder, host):
    """
     Not transfer already transfered tables


    """
    ssh = util_aws.aws_ec2_ssh(hostname=host)
    ssh.put_all(fromfolder, tofolder)



def batch_launch_remote() :
    """
     batcher launch
     monitor launch

     Will put as starter in batch system...


   
    """
    ssh = util_aws.aws_ec2_ssh(hostname=host)
    ssh.cmd("batch_daemon_launchi_cli.py  --task_folder ")




def batch_result_retrieve(folder_remote, folder_local, host):
  """"
      dont retrieve existing folder on disk.


  """"
  ssh = util_aws.aws_ec2_ssh(hostname=host)
  ssh.get_all(fromfolder, tofolder)





################################################################################
def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do",  default="launch_ec2", help="retrieve_fromec2")
    parser.add_argument("--folder_local", help="Select path for a .csv  HyperParameters ")
    parser.add_argument("--folder_remote", default=FOLDER_DEFAULT, help="Absolute / relative path to working folder.")
    parser.add_argument("--host", default="34.67.74.78", help="Host name")
    parser.add_argument("--cmd_remote", default="main.py", help="Name of the main script")
    parser.add_argument("--log_file", default="log_aws_inout.log", help=".")

    options = parser.parse_args()
    return options




################################################################################
if __name__ == '__main__':
    args   = load_arguments()
    # APP_ID = util_log.create_appid(__file__)
    logger = util_log.logger_setup(__name__,
                                   log_file= args.log_file,
                                   formatter= util_log.FORMATTER_4)

    if args.do == "launch_ec2" :
       pass



    if args.do == "get_fromec2" :
       pass

    if args.do == "put_toec2" :
       log( "Current Process Id:", os.getpid()  )
       valid_task_folders = get_list_valid_task_folder(args.task_folder)
       log("All task Completed")






"""



DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD+'/aapackage');  sys.path.append(DIRCWD+'/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
print 'Directory Folder', DIRCWD
###################################################################################################
EC2CWD=      '/home/ubuntu/notebook/'
EC2_ipython= '/home/ubuntu/anaconda2/bin/'


####  EC2 + Task  #################################################################################
task1_name= 'elvis_prod_20160102'



'''  Manual Input
ec2_id=      'i-000155755737d04aa'
host=        '52.78.74.143'                  #Elastic IP
task1_name=  'elvis_prod_20161228'
#     host=   '52.79.79.1'
'''





##################################################################################################
DIRBATCH= DIRCWD + '/linux/batch/task/' + task1_name
execfile(DIRBATCH + '/ec2_config.py')
print 'ec2_id', ec2_id,  ', host', host, ', task1_name',  task1_name

ssh= util.aws_ec2_ssh(host)
con=  util.aws_conn_create(region="ap-northeast-2"); con

##################################################################################################
#### Retrieve Results from EC2 from output folder   ##############################################
ssh.get_all( EC2CWD + '/linux/batch/output/',  DIRCWD +'/zdisks3/results/'+task1_name)



#### Loop to retrieve results   ##################################################################
import time; ii=0;  ipython_idle_count=0
while True :
   ii+=1
   print ' \n\n', ii, util.date_nowtime() + '\n'

   ssh.get_all( EC2CWD + '/linux/batch/output/',  DIRCWD +'/zdisks3/results/'+task1_name)

   util.os_zipfolder(DIRCWD +'/zdisks3/results/'+task1_name+'/output/',         #Zip for Backup
                     DIRCWD +'/zdisks3/results/'+task1_name+'/output.zip')

   #Check if no CPU is running ipython (== idle)
   aux= ssh.cmd2('pgrep ipython')
   if aux[0][0] == '' : ipython_idle_count+= 1
   else :               ipython_idle_count= 0
   print 'ipython_idle_count', ipython_idle_count

   if ipython_idle_count > 1 :                   #2*5min, 10 mins Close the Instance
      con=  util.aws_conn_create(region="ap-northeast-2")
      con.terminate_instances(instance_ids=[ec2_id])
      con.release_address(public_ip=host)
      # con.stop_instances(instance_ids=[ec2_id])
      sys.exit(0)

   time.sleep( 8 * 60)  # Retrieve every 5mins




"""
