# -*- coding: utf-8 -*-
'''
  Daemon for auto-scale.
  Only launch in master instance - Some identification so that this scripts silently exits.
  
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
from aapackage import util_aws


############### logger ########################################################
logger = None
keypair = 'ec2_linux_instance'
region = 'us-west-2'  # Oregon West
amiId = 'ami-0491a657e7ed60af7'
spot_cfg_file = '/tmp/ec_spot_config'


################################################################################
def log(*argv):
  logger.info(",".join([str(x) for x in argv]))


################################################################################
def get_list_valid_task_folder(folder, script_regex=r'main\.(sh|py)'):
  """ Make it regex based so that both shell and python can be checked. """
  if not os.path.isdir(folder):
      return []
  valid_folders = []
  for root, dirs, files in os.walk(folder):
      root_splits = root.split("/")
      for filename in files:
          if "_qstart" not in root_splits[-1] and 
              "_qdone" not in root_splits[-1] and 
              re.match(filename, script_regex, re.I) and 
              "_ignore" not in root_splits[-1]  :
                  valid_folders.append(root)
  return valid_folders


################################################################################
def load_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task_folder", default=TASK_FOLDER_DEFAULT, help="path to task folder.")
  parser.add_argument("--log_file", default="batchdaemon_autoscale.log", help=".")
  parser.add_argument("--mode", default="nodaemon", help="daemon/ .")
  parser.add_argument("--waitsec", type=int, default=60, help="wait sec")
  parser.add_argument("--max_instance", type=int, default=2, help="")
  parser.add_argument("--max_cpu", type=int, default=16, help="")  
  options = parser.parse_args()
  return options


################################################################################
def start_rule(nb_task_remaining=0, nb_CPU_available=10):
  """ Start spot instance if more than 10 tasks or less than 10 CPUs """
  return nb_task_remaining > 10 and nb_CPU_available < 10

################################################################################
def stop_rule(nb_task_remaining=0):
  """ Stop rule for terminating the spot instances. """    
  return nb_task_remaining == 0


################################################################################
def ec2_instance_usage(instance_id=None, ipadress=None):
  """
  https://stackoverflow.com/questions/20693089/get-cpu-usage-via-ssh
  https://haloseeker.com/5-commands-to-check-memory-usage-on-linux-via-ssh/
  """
  pass


def build_template_config(instance_type):
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
  with open(spot_cfg_file, 'w') as spot_file:
    spot_file(json.dumps(spot_config))


################################################################################
def ec2_spot_start(instance_type, spot_price):
  """
  Request a spot instance based on the price for the instance type
  # Need a check if this request has been successful.
  """
  if not instance_type:
    instance_type = 't3.small'
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
  ll= ec2_instance_getlist()


################################################################################
def ec2_instance_stop(instance_list) :
  """ Stop the spot instances as u stop any other instance, this should work"""
  instances = instance_list
  if instances:
    if isinstance(instance_list, list)
        instances = ','.join(instance_list)
    cmdargs = [
      'aws', 'ec2', 'stop-instances', 
      '--instance-ids', instances
    ]
    cmd = ' '.join(cmdargs)
    os.system(cmd)


################################################################################
def ec2_instance_backup(instance_list, folder_list=["/zlog/"]) :
    """
      zip some local folders
      Tansfer data from local to /zs3drive/backup/AMIname_YYYYMMDDss/
    
      Transfer task_out/  to /zs3drive/task_out/
    
    """
    pass


#################################################################################
if __name__ == '__main__':
  args   = load_arguments()
  logging.basicConfig()
  logger = logger_setup(__name__, log_file=args.log_file,
                        formatter=util_log.FORMATTER_4, isrotate  = True)
  instances_dict = {"id" :{ "ncpu":0, "ip_adress": "" }  }
  log("Daemon start: ", os.getpid())
  while True:
    log("Daemon new loop: ", args.task_folder)
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





