# -*- coding: utf-8 -*-
'''
  Daemon for auto-scale.
  Only launch in master instance 
  - Some identification so that this scripts silently exits.
  
### S3 does NOT support folder rename, bash shell to replance rename  
  rename() {
  #do things with parameters like $1 such as
  cp  $1   $2  --recursive  && rm $1     --recursive
}

  
it takes 3ms to read+write task_list
2019-04-19 13:59:58,587, 12599, batch_daemon_launch_cli.py,    0.031108617782592773


  
  
  Auto-Scale :  
    batch_daemon_autoscale_cli.py(ONLY on master instance) - how to check this ?
    Start Rule:
      nb_task_remaining > 10 AND nb_CPU_available < 10 
        start new spot Instance by AWS CLI from spot template
    Stop Rule:
      nb_task_remaining = 0 for last 5mins : 
        stop instance by AWS CLI.
    keypair: ec2_linux_instance
    Oregon West - us-west-2
    AMI :  ami-0491a657e7ed60af7
    Instance spot :  t3.small
    Common Drive is Task Folders: /home/ubuntu/zs3drive/tasks/
    Out for tasks : /home/ubuntu/zs3drive/tasks_out/

keypair = 'ec2_linux_instance'
region = 'us-west-2'  # Oregon West
amiId = 'ami-0491a657e7ed60af7'
instance_type = 't3.small'
spot_price = '0.55'
cmdargs = [
  'aws', 'ec2', 'request-spot-instances',
  '--region', region,
  '--spot-price', spot_price,
  '--instance-count', 1,
  ' --type', 'one-time',
  '--launch-specification', '/tmp/ec_spot_config.json'
]
cmd = ' '.join(cmdargs)

spot_config = {
  "ImageId": amiId,
  "KeyName": keypair, 
  "SecurityGroupIds": ["sg-4b1d6631", "sg-42e59e38"],        
  "InstanceType": instance_type,
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
with open('/tmp/ec_spot_config', 'w') as spot_file:
  spot_file(json.dumps(spot_config))

'''
#################################################################################
import json
import re
import os
import sys
from time import sleep
import argparse
import logging
import subprocess


################################################################################
from aapackage.util_log import logger_setup
from aapackage.batch import util_cpu
from aapackage.util_aws import aws_ec2_ssh
from aapackage import util_log



############### logger ########################################################
logger = None
TASK_FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__)) + "/ztestasks/"
keypair = 'ec2_linux_instance'
region  = 'us-west-2'  # Oregon West
default_instance_type = 't3.small'
amiId = 'ami-0491a657e7ed60af7'
spot_cfg_file = '/tmp/ec_spot_config'


### Maintain infos on all instances  ###########################################
global instance_dict
instances_dict = {"id" :{  "ncpu":0, "ip_address": "", 'ram':0, 'cpu_usage': 0, 'ram_usage':0 }  }


### Record the running/done tasks on S3 DRIVE, Global File system  #############
global_task_file = "/home/ubuntu/zs3drive/global_task.json"



################################################################################
def log(*argv):
  logger.info(",".join([str(x) for x in argv]))



################################################################################
def load_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_file", default="batchdaemon_autoscale.log", help=".")
  parser.add_argument("--mode", default="nodaemon", help="daemon/ .")

  parser.add_argument("--global_task_file", default=global_task_file, help="global task file")    
  parser.add_argument("--task_folder", default=TASK_FOLDER_DEFAULT, help="path to task folder.")  
  parser.add_argument("--instance", default=default_instance_type, help="Type of soot instance")
  parser.add_argument("--spotprice", type=float, default=0.0, help="Actual price offered by us.")
  parser.add_argument("--waitsec", type=int, default=60, help="wait sec")
  parser.add_argument("--max_instance", type=int, default=2, help="")
  parser.add_argument("--max_cpu", type=int, default=16, help="")  

  options = parser.parse_args()
  return options




################################################################################
def task_get_list_valid_folder(folder, script_regex=r'main\.(sh|py)'):
  """ Make it regex based so that both shell and python can be checked. """
  if not os.path.isdir(folder):
      return []
  valid_folders = []
  for root, dirs, files in os.walk(folder):
      root_splits = root.split("/")
      for filename in files:
          if "_qstart" not in root_splits[-1] and  \
              "_qdone" not in root_splits[-1] and  \
              re.match(filename, script_regex, re.I) and  "_ignore" not in root_splits[-1]  :
                  valid_folders.append(root)
  return valid_folders


def task_get_list_valid_folder_new(folder_main):
  """ Why was this added  /
    --->  S3 disk drive /zs3drive/  DOES NOT SUPPORT FOLDER RENAMING !! due to S3 limitation.
    --->  Solution is to have a Global File global_task_dict which maintains current "running tasks/done tasks"
          Different logic should be applied ,  see code in batch_daemon_launch.py
  
  """
  folder_check = json.load(open(global_task_file, mode="r")) 
  task_started = {k  for k, _ in folder_check  }
  task_all = {x for x in os.listdir(folder_main) if os.path.isdir(x)  }    
  return list( task_all.difference(task_started) ) 
  

def task_isvalid_folder(folder_main, folder, folder_check, global_task_file):
  if os.path.isfile(os.path.join(folder_main, folder)) or folder in folder_check :
     return False
  elif "_qdone" in folder  or "_qstart" in folder or "_ignore" in folder    :
     # global_task_file_save(folder, folder_check, global_task_file) 
     return False
  else :  
     return True

    
def task_getcount(folder_main):
  """ Number of tasks remaining to be scheduled for run """
  ###task already started
  folder_check = json.load(open(global_task_file, mode="r")) 
  task_started = {k  for k, _ in folder_check.items()  }
  task_all = {x for x in os.listdir(folder_main) if os.path.isdir(x)  }    

  return len(task_all.difference(task_started))


  
################################################################################
def instance_start_rule( task_folder):
  """ Start spot instance if more than 10 tasks or less than 10 CPUs 
      return instance type, spotprice
  """
  global instance_dict
  ntask = task_getcount(task_folder)
  ncpu  = instance_get_ncpu(instances_dict)
  
  if  ntask > 30 and ncpu < 10 :
    return {'type' : 't3.medium', 'spotprice' : 0.25}
  
  elif  ntask > 10 and ncpu < 5 :
    return {'type' : 't3.small', 'spotprice' : 0.25}  
  else :
    return None
   
  
def instance_stop_rule( task_folder):
  """IF spot instance usage is ZERO CPU%  and RAM is low --> close instances."""
  global instance_dict
  ntask         = task_getcount(task_folder)
  instance_dict =  ec2_instance_getallstate()
  if ntask == 0 and  instances_dict :
      # Idle Instances
      instance_list = [k for k,x in instances_dict.items() if x["cpu_usage"] < 5.0 and x["ram_usage"] < 10.0]
      return instances_list
  else :
      return None
   
   
def instance_get_ncpu(instances_dict):
  """ Total cpu count for the launched instances. """
  ss = 0
  if instances_dict :
   for x in instances_dict.items() :
     ss += x["cpu"]
  return ss
   
  
def ec2_instance_getallstate():
  """
      use to update the global instance_dict
          "id" :
          instance_type,
          ip_address
          ncpu, ram,
          cpu_usage, ram_usage
  """
  pass
  

  
################################################################################ 
def ec2_instance_usage(instance_id=None, ipadress=None):
  """
  https://stackoverflow.com/questions/20693089/get-cpu-usage-via-ssh
  https://haloseeker.com/5-commands-to-check-memory-usage-on-linux-via-ssh/
  """
  if instance_id and ipadress:
    ssh = aws_ec2_ssh(hostname=ipadress)
    cmdstr = "top -b -n 10 -d.2 | grep 'Cpu' | awk 'NR==3{ print($2)}'"
    ssh.cmd(cmdstr)


def build_template_config(instance_type):
  """ Build the spot json config into a json file. """
  spot_config = {
    "ImageId": amiId,
    "KeyName": keypair, 
    "SecurityGroupIds": ["sg-4b1d6631", "sg-42e59e38"],        
    "InstanceType": instance_type if instance_type else default_instance_type,
    "IamInstanceProfile": {
      "Arn": "arn:aws:iam::013584577149:instance-profile/ecsInstanceRole"
    },
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/instance1",
        "Ebs": {
          "DeleteOnTermination": true,
          "VolumeSize": 60
        }                      
      }
    ]
  }
  with open(spot_cfg_file, 'w') as spot_file:
    spot_file(json.dumps(spot_config))


################################################################################
def ec2_spot_start(instance_type, spot_price):
  """
  Request a spot instance based on the price for the instance type
  # Need a check if this request has been successful.
  """
  if not instance_type:
    instance_type = default_instance_type
  build_template_config(instance_type)  
  cmdargs = [
    'aws', 'ec2', 'request-spot-instances',
    '--region', region,
    '--spot-price', spot_price,
    '--instance-count', 1,
    ' --type', 'one-time',
    '--launch-specification', spot_cfg_file
  ]
  cmd = ' '.join(cmdargs)
  # ss  = 'aws ec2 request-spot-instances   --region us-west-2  --spot-price "0.55" --instance-count 1 '
  # ss += ' --type "one-time" --launch-specification "file://ec2_spot_t3small.json" '
  msg = os.system(cmd)
  ll= ec2_spot_instance_list()
  return instance_list['SpotInstanceRequests'] if 'SpotInstanceRequests' in ll else []


def ec2_spot_instance_list():
  """ Get the list of current spot instances. """
  cmdargs = [
    'aws', 'ec2', 'describe-spot-instance-requests'
  ]
  cmd = ' '.join(cmdargs)
  value = os.popen(cmd).read()
  try:
    instance_list = json.loads(value)
  except:
    instance_list = {
      "SpotInstanceRequests": []
    }
  return instance_list
  

def ec2_instance_stop(instance_list) :
  """ Stop the spot instances ainstances u stop any other instance, this should work"""
  instances = instance_list
  if instances:
    if isinstance(instance_list, list) :
        instances = ','.join(instance_list)
    cmdargs = [
      'aws', 'ec2', 'stop-instances', 
      '--instance-ids', instances
    ]
    cmd = ' '.join(cmdargs)
    os.system(cmd)
    return instances.split(",")

  
def ec2_instance_backup(instance_list, folder_list=["/zlog/"]) :
    """
      zip some local folders
      Tansfer data from local to /zs3drive/backup/AMIname_YYYYMMDDss/
    
    """
    pass


  
  
#################################################################################
if __name__ == '__main__':
  args   = load_arguments()
  # logging.basicConfig()
  logger = logger_setup(__name__, log_file=args.log_file,
                        formatter=util_log.FORMATTER_4, isrotate  = True)
  
  global_task_file = args.global_task_file
  
  log("Daemon start: ", os.getpid(), global_task_file)
  while True:
    log("Daemon new loop: ", args.task_folder)
    
    # Keep Global state of running instances
    instances_dict =  ec2_instance_getallstate()
    
    
    ### Start instance by rules
    start_instance = instance_start_rule( args.task_folder)
    if start_instance : 
        # When instance start, batchdaemon will start and picks up task in  COMMON DRIVE /zs3drive/
        instance_list = ec2_spot_start(start_instance['type'], start_instance['spotprice']  )
        log("Starting instances", instance_list)
        sleep(30)
    
    
    ### Stop instance by rules
    stop_instances = instance_stop_rule( args.task_folder)
    if stop_instances:
      ec2_instance_backup(stop_instances, folder_list=[ "/home/ubuntu/zlog/"])
      ec2_instance_stop(stop_instances)
      log("Stopped instances", stop_instances)

    if args.mode != "daemon":
      log("Daemon","terminated", os.getpid())
      break

    sleep(args.waitsec)

    
    
    
    
