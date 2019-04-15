# -*- coding: utf-8 -*-
'''
  Daemon for auto-scale.
  Only launch in master instance.
  
  
  Auto-Scale :  
   batch_daemon_autoscale_cli.py (ONLY on master instance)
   Some basic rules like :
       nb_task_remaining > 10    AND   nb_CPU_available < 10 
          start new spot Instance  by AWS CLI. from spot template


       nb_task_remaining = 0 for last 5mins : 
          stop instance  by AWS CLI.
         
  

ec2_linux_instance


ss  = 'aws ec2 request-spot-instances   --region us-west-2  --spot-price "0.55" --instance-count 1 '
ss += ' --type "one-time" --launch-specification "file://D:\_devs\Python01\\awsdoc\\ec_config2.json" '


{
  "ImageId": "ami-0b01d9c82fc6d1391",
  "KeyName": "ec2_linux_instance", 
  "SecurityGroupIds": [ "sg-4b1d6631"  ,   "sg-42e59e38"  ], 
          
          
  "InstanceType": "",
  



"IamInstanceProfile": {
                        "Arn": "arn:aws:iam::013584577149:instance-profile/ecsInstanceRole"
},


                    

"BlockDeviceMappings": [
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "DeleteOnTermination": true,
                                "VolumeSize": 60
                            }
                            
                            
                            
                        }
                    ]
                    
}








util_aws
  

Oregon West
You can use this AMI :  ami-0491a657e7ed60af7
Instance spot :  t3.small




Common Drive is Task Folders is 
    /home/ubuntu/zs3drive/tasks/

Out for tasks :
    /home/ubuntu/zs3drive/tasks_out/






'''
import os
import sys
from time import sleep
import argparse
import logging
import subprocess

################################################################################
from aapackage import util_log
from aapackage.batch import util_cpu
from aapackage import util_aws



############### logger ########################################################
global logger
logger = logging.basicConfig()

def log(*argv):
  logger.info(",".join([str(x) for x in argv]))


################################################################################
def load_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task_folder",       default=TASK_FOLDER_DEFAULT, help="path to task folder.")
  parser.add_argument("--log_file",          default="batchdaemon_autoscale.log", help=".")
  parser.add_argument("--mode",              default="nodaemon", help="daemon/ .")
  parser.add_argument("--waitsec",      type=int, default=60, help="wait sec")
  parser.add_argument("--max_instance", type=int, default=2, help="")
  parser.add_argument("--max_cpu",      type=int, default=16, help="")  
  options = parser.parse_args()
  return options


################################################################################
def start_rule() :
    pass


def stop_rule() :
    pass



def ec2_instance_usage(instance_id=None, ipadress=None):
    """
    https://stackoverflow.com/questions/20693089/get-cpu-usage-via-ssh
    
    https://haloseeker.com/5-commands-to-check-memory-usage-on-linux-via-ssh/
    
    """
    
    

def ec2_spot_start(instance_type ):
    """
     

    
    """
    ss  = 'aws ec2 request-spot-instances   --region us-west-2  --spot-price "0.55" --instance-count 1 '
    ss += ' --type "one-time" --launch-specification "file://ec2_spot_t3small.json" '
    
    msg = os.system(ss )

    ll= ec2_instance_getlist()



def ec2_instance_stop(instance_list) :
    pass



def ec2_instance_backup(instance_list, folder_list=["/zlog/"]) :
    """
      zip some local folders
      Tansfer data from local to /zs3drive/backup/AMIname_YYYYMMDDss/
    
      Transfer task_out/  to /zs3drive/task_out/
    
    """
    pass




#################################################################################
#################################################################################
if __name__ == '__main__':
    args   = load_arguments()
    logger = util_log.logger_setup(__name__,
                                   log_file  = args.log_file,
                                   formatter = util_log.FORMATTER_4,
                                   isrotate  = True)


   instances_dict = {"id" :{ "ncpu":0, "ip_adress": "" }  }
   

   log("Daemon","start ", os.getpid())
   while True:
    log("Daemon new loop", args.task_folder)
    folders = get_list_valid_task_folder(args.task_folder)
    ntask   = len(folders)


    ### Start instance by rules
    instance_type = start_rule(ntask, instances_dict)
    if instance_type : 
        ## When instance start, batchdaemon will start and picks up task in COMMON DRIVE /zs3drive/
        ec2_spot_start(instance_type )
  
  
    ### Stop instance by rules
    instance_list = stop_rule(ntask, instances_dict)
    ec2_instance_backup(  instance_list, folder_list=[ "/home/ubuntu/zlog/"])
    ec2_instance_stop(  instance_list)
  
  

    if args.mode != "daemon":
      log("Daemon","terminated", os.getpid())
      break

    sleep(args.waitsec)





