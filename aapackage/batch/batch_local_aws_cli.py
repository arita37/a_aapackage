# -*- coding: utf-8 -*-
"""
Many tasks :

Launch AWS server
Send task folder from LOCAL to remote AWS
Retrieve results from AWS folder to local.
Filter out already retrieve results.
Start on AWS the batch_daemon_launchi_cli.py



"""
import os
import sys
from time import sleep
import argparse
import logging

################################################################################
from aapackage import util_log, util_aws
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
    pass





def task_transfer_to_ec2(fromfolder, tofolder, host):
    ssh = util_aws.aws_ec2_ssh(hostname=host)
    ssh.put_all(fromfolder, tofolder)



def batch_launch_remote() :
    ssh = util_aws.aws_ec2_ssh(hostname=host)
    ssh.cmd("python ")




def batch_result_retrieve(folder_remote, folder_local, host):
  pass





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








