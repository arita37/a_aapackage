# -*- coding: utf-8 -*-
'''
## 
cd aapackage
pip installl -e .



##### Daemon mode
batch_daemon_autoscale_cli.py --mode daemon --task_folder  zs3drive/tasks/  --log_file zlog/batchautoscale.log   



#### Test with reset task file, on S3 drive
batch_daemon_autoscale_cli.py --task_folder  zs3drive/tasks/  --log_file zlog/batchautoscale.log   --reset_global_task_file 1




  Daemon for auto-scale.
  Only launch in master instance 
  - Some identification so that this scripts silently exits.
  
  ### S3 does NOT support folder rename, bash shell to replance rename
  rename() {
    #do things with parameters like $1 such as
    cp  $1   $2  --recursive  && rm $1     --recursive
  }

  it takes 3ms to read+write task_list
  2019-04-19 13:59:58,587, 12599, batch_daemon_launch_cli.py, 0.031108617782592773


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
import paramiko


################################################################################
from aapackage.util_log import logger_setup
from aapackage.batch import util_cpu
from aapackage.util_aws import aws_ec2_ssh
from aapackage import util_log



############### logger ########################################################
ISTEST = True   ### For test the code

logger = None
TASK_FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__)) + "/ztest/tasks/"
#TASK_FOLDER_DEFAULT =  "/home/ubuntu/ztest/tasks/"

keypair = 'aws_ec2_ajey'
region  = 'us-west-2'  # Oregon West
default_instance_type = 't3.small'
amiId = "ami-04b010cceb44affce"  #'ami-0491a657e7ed60af7'
spot_cfg_file = '/tmp/ec_spot_config'



### Maintain infos on all instances  ###########################################
# global instance_dict
instance_dict = {
  "id":{
    "id": "",
    "cpu": 0,
    "ip_address": "",
    'ram': 0,
    'cpu_usage': 0,
    'ram_usage':0
  }
}


### Record the running/done tasks on S3 DRIVE, Global File system  #############
global_task_file = "%s/zs3drive/global_task.json" % (os.environ['HOME'] 
                     if 'HOME' in os.environ else '/home/ubuntu')


################################################################################
def log(*argv):
  logger.info(",".join([str(x) for x in argv]))


################################################################################
def load_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_file", default="batchdaemon_autoscale.log",  help=".")
  parser.add_argument("--mode", default="nodaemon", help="daemon/ .")
  parser.add_argument("--global_task_file", default=global_task_file, help="global task file")
  parser.add_argument("--task_folder", default=TASK_FOLDER_DEFAULT, help="path to task folder.")

  parser.add_argument("--reset_global_task_file", default=0, help="global task file Reset File")


  parser.add_argument("--ami", default=amiId,   help="AMI used for spot")
  
  parser.add_argument("--instance", default=default_instance_type,   help="Type of soot instance")
  parser.add_argument("--spotprice", type=float, default=0.0, help="Actual price offered by us.")
  parser.add_argument("--waitsec", type=int, default=60, help="wait sec")
  parser.add_argument("--max_instance", type=int, default=2, help="")
  parser.add_argument("--max_cpu", type=int, default=16, help="")  
  options = parser.parse_args()
  return options




################################################################################
def task_get_list_valid_folder(folder, script_regex=r'main\.(sh|py)'):
  """ Make it regex based so that both shell and python can be checked. 
       _qstart, _ignore , _qdone are excluded.
       main.sh or main.py should be in the folder.
  
  """
  if not os.path.isdir(folder):
    return []
  valid_folders = []
  for root, dirs, files in os.walk(folder):
    root_splits = root.split("/")
    for filename in files:
      if re.match(script_regex, filename, re.I) and \
        not re.match(r'^.*(_qstart|_qdone|_ignore)$', root_splits[-1], re.I):
        valid_folders.append(root)
  return valid_folders


def task_get_list_valid_folder_new(folder_main):
  """ Why was this added  /
    --->  S3 disk drive /zs3drive/  DOES NOT SUPPORT FOLDER RENAMING !! due to S3 limitation.
    --->  Solution is to have a Global File global_task_dict which maintains current "running tasks/done tasks"
          Different logic should be applied ,  see code in batch_daemon_launch.py
  
  """
  # task already started
  folder_check = json.load(open(global_task_file, mode="r")) 
  task_started = {k for k in folder_check}
  # There could be problem here, if none of them is a directory, so it
  # becomes a dict, difference  betn a set and dict will not work.
  task_all = {x for x in os.listdir(folder_main) if os.path.isdir('%s/%s' % (folder_main, x))}
  folders = list(task_all.difference(task_started))
  valid_folders = []
  for folder in folders:
    if task_isvalid_folder(folder_main, folder, folder_check):
      valid_folders.append(folder)
      
  print(valid_folders)    
  return valid_folders


def task_isvalid_folder(folder_main, folder, folder_check):
  # Invalid cases
  if os.path.isfile(os.path.join(folder_main, folder)) or \
    folder in folder_check or \
    re.search(r'_qstart|_qdone|_ignore', folder, re.I):
    return False
  else:
     #Valid case
     return True


def task_getcount(folder_main):
  """ Number of tasks remaining to be scheduled for run """
  return len(task_get_list_valid_folder_new(folder_main))


##########################################################################################
def get_spot_price(instance_type):
  """ Get the spot price for instance type in us-west-2"""
  value = 0.0
  if os.path.exists('./aws_spot_price.sh') and os.path.isfile('./aws_spot_price.sh'):
    cmdstr = "./aws_spot_price.sh %s | grep Price | awk '{print $2}'" % instance_type
    value = os.popen(cmdstr).read()
    value = value.replace('\n', '') if value else 0.10
    #parsefloat(value)
  return parsefloat(value)


def parsefloat(value, default=0.0):
  """ Parse the float value. """
  fltvalue = default
  try:
    fltvalue = float(value)
  except:
    pass
  return fltvalue




################################################################################
def instance_get_ncpu(instances_dict):
  """ Total cpu count for the launched instances. """
  ss = 0
  if instances_dict :
   for _, x in instances_dict.items() :
     ss += x["cpu"]
  return ss


################################################################################
def ec2_instance_getallstate():
  """
      use to update the global instance_dict
          "id" :
          instance_type,
          ip_address
          cpu, ram,
          cpu_usage, ram_usage
  """
  val = {}
  spot_list = ec2_spot_instance_list()
  spot_instances = []
  for spot_instance in spot_list['SpotInstanceRequests']:
    if re.match(spot_instance['State'], 'active', re.I) and \
      'InstanceId' in spot_instance:
        spot_instances.append(spot_instance['InstanceId'])
  # print(spot_instances)
  
  for spot in spot_instances:
    cmdargs = ['aws', 'ec2', 'describe-instances', '--instance-id', spot]
    cmd = ' '.join(cmdargs)
    value = os.popen(cmd).read()
    inst = json.loads(value)
    ncpu = 0
    ipaddr = None
    instance_type = default_instance_type
    if inst and 'Reservations' in inst and inst['Reservations']:
      reserves = inst['Reservations'][0]
      if 'Instances' in reserves and reserves['Instances']:
        instance = reserves['Instances'][0]
        
        if 'CpuOptions' in instance and 'CoreCount' in instance['CpuOptions']:
          ncpu = instance['CpuOptions']['CoreCount']
          
        if 'PublicIpAddress' in instance and instance['PublicIpAddress']:
          ipaddr = instance['PublicIpAddress']
        
        instance_type = instance['InstanceType']
    
    if ipaddr:
      cpuusage, usageram, totalram = ec2_instance_usage(spot, ipaddr)
      # print(cpuusage, usageram, totalram)
      val[spot] = {
        'id': spot,
        'instance_type': instance_type,
        'cpu': ncpu,
        'ip_address': ipaddr,
        'ram': totalram,
        'cpu_usage': cpuusage,
        'ram_usage': usageram
      }
  # print(val)
  return val


################################################################################
def run_command_thru_ssh(hostname, key_file, cmdstr, remove_newline=True):
  """ Make an ssh connection using paramiko and  run the command"""
  try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, key_filename=key_file, timeout=5)
    stdin, stdout, stderr = ssh.exec_command(cmdstr)
    data = stdout.readlines()
    if remove_newline:
      value = ''.join(data).replace('\n', '')
    else:
      value = ''.join(data)
    ssh.close()
  except:
    value = None
  return value


def ec2_keypair_get():
  identity = "%s/.ssh/%s" % \
            (os.environ['HOME'] if 'HOME' in os.environ else '/home/ubuntu', keypair)
  return identity


################################################################################ 
def ec2_instance_usage(instance_id=None, ipadress=None):
  """
  https://stackoverflow.com/questions/20693089/get-cpu-usage-via-ssh
  https://haloseeker.com/5-commands-to-check-memory-usage-on-linux-via-ssh/
  """
  cpuusage = None
  ramusage = None
  totalram = None
  if instance_id and ipadress:
    identity = ec2_keypair_get()
    # ssh = aws_ec2_ssh(hostname=ipadress, key_file=identity)
    # cmdstr = "top -b -n 10 -d.2 | grep 'Cpu' | awk 'NR==3{ print($2)}'"
    cmdstr = "top -b -n 10 -d.2 | grep 'Cpu' | awk 'BEGIN{val=0.0}{ if( $2 > val ) val = $2} END{print(val)}'"
    # cpu = ssh.command(cmdstr)
    cpuusage = run_command_thru_ssh(ipadress, identity, cmdstr)
    cpuusage = 100.0  if not cpuusage else float(cpuusage)


    cmdstr = "free | grep Mem | awk '{print $3/$2 * 100.0, $2}'"
    # ram = ssh.command(cmdstr)
    ramusage = run_command_thru_ssh(ipadress, identity, cmdstr)
    
    if not ramusage:
        totalram = 0
        usageram = 100.0
    else:
        vals = ramusage.split()
        usageram = float(vals[0]) if vals and vals[0] else 100.0
        totalram = int(vals[1]) if vals and vals[1] else 0
    
  return cpuusage, usageram, totalram


################################################################################
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
        "DeviceName": "/dev/sda1",
        "Ebs": {
          "DeleteOnTermination": True,
          "VolumeSize": 60
        }                      
      }
    ]
  }
  with open(spot_cfg_file, 'w') as spot_file:
    spot_file.write(json.dumps(spot_config))


################################################################################
def ec2_spot_start(instance_type, spot_price, waitsec=100):
  """
  Request a spot instance based on the price for the instance type
  # Need a check if this request has been successful.
  
  100 sec to be provisionned and started.
  """
  if not instance_type:
    instance_type = default_instance_type
  build_template_config(instance_type)  
  cmdargs = [
    'aws', 'ec2', 'request-spot-instances',
    '--region', region,
    '--spot-price', str(spot_price),
    '--instance-count', "1",
    ' --type', 'one-time',
    '--launch-specification', 'file://%s' % spot_cfg_file
  ]
  print(cmdargs)
  cmd = ' '.join(cmdargs)
  msg = os.system(cmd)
  sleep(waitsec)  # It may not be fulfilled in 50 secs.
  ll= ec2_spot_instance_list()
  return ll['SpotInstanceRequests'] if 'SpotInstanceRequests' in ll else []



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


################################################################################
def ec2_instance_stop(instance_list) :
  """ Stop the spot instances ainstances u stop any other instance, this should work"""
  instances = instance_list
  if instances:
    if isinstance(instance_list, list) :
        instances = ','.join(instance_list)
    cmdargs = [
      'aws', 'ec2', 'terminate-instances',
      '--instance-ids', instances
    ]
    cmd = ' '.join(cmdargs)
    os.system(cmd)
    return instances.split(",")


################################################################################
def ec2_instance_backup(instances_list, folder_list=["zlog/"],
                        folder_backup="/home/ubuntu/zs3drive/backup/") :
    """
      Zip some local folders
      Tansfer data from local to /zs3drive/backup/AMIname_YYYYMMDDss/
    tar -czvf directorios.tar.gz folder
    
    """
    from datetime import datetime
    now = datetime.today().strftime('%Y%m%d')
    for inst in instances_list :
      ssh = aws_ec2_ssh( inst["ip_address"])
      target_folder = folder_backup + inst["id"] +  "_" + now
      cmdstr = "mkdir %s" % target_folder
      print(cmdstr)
      ssh.cmd( "mkdir " + target_folder )
      for t in folder_list :
        cmdstr = "tar -czvf  %s/%s.tar.gz %s" % (target_folder,
                                                 t.replace('/', ''), t)
        print(cmdstr)
        ssh.cmd(cmdstr)


################################################################################
def instance_start_rule(task_folder):
  """ Start spot instance if more than 10 tasks or less than 10 CPUs 
      return instance type, spotprice
  """
  global instance_dict
  ntask = task_getcount(task_folder)
  ncpu  = instance_get_ncpu(instance_dict)
  print("Ntask, ncou", ntask, ncpu)
  
  if ntask == 0  and not ISTEST :
    return None
    
  # hard coded values here
  if  ntask > 30 and ncpu < 10 :
    # spotprice = max(0.05, get_spot_price('t3.medium')* 1.30)
    spotprice = 0.05
    return {'type' : 't3.medium', 'spotprice' : spotprice  }
    

  if  ntask > 20 and ncpu < 5 :
    # spotprice = max(0.05, get_spot_price('t3.medium')* 1.30)
    spotprice = 0.05
    return {'type' : 't3.small', 'spotprice' : spotprice }
  
  
  if  ntask > 2 and ncpu < 5 :
    # spotprice = max(0.05, get_spot_price('t3.medium')* 1.30)
    # spotprice = 0.05
    # return {'type' : 't3.small', 'spotprice' : spotprice }
    spotprice = 0.05
    return {'type' : 't3.medium', 'spotprice' : spotprice }  
    
  return None


def instance_stop_rule(task_folder):
  """IF spot instance usage is ZERO CPU%  and RAM is low --> close instances."""
  global instance_dict
  ntask = task_getcount(task_folder)
  instance_dict = ec2_instance_getallstate()
  if ntask == 0 and  instance_dict :
      # Idle Instances
      instance_list = [x for _, x in instance_dict.items() if x["cpu_usage"] < 5.0 and x["ram_usage"] < 10.0]
      return instance_list
  else :
      return None




##########################################################################################
if __name__ == '__main__':
  args   = load_arguments()
  # logging.basicConfig()
  logger = logger_setup(__name__, log_file=args.log_file,
                        formatter=util_log.FORMATTER_4, isrotate  = True)
  
  global_task_file = args.global_task_file
  key_file = ec2_keypair_get()


  if args.reset_global_task_file :
    with open(global_task_file, 'w') as f:
       json.dump({}, f)    


  log("Daemon start: ", os.getpid(), global_task_file)
  while True:
    log("Daemon new loop: ", args.task_folder)
    
    # Keep Global state of running instances
    instance_dict =  ec2_instance_getallstate()
    
    
    ### Start instance by rules ###############################################
    start_instance = instance_start_rule( args.task_folder)
    log("Starting instances", start_instance)
    if start_instance : 
        # When instance start, batchdaemon will start and picks up task in  COMMON DRIVE /zs3drive/
        instance_list = ec2_spot_start(start_instance['type'], start_instance['spotprice']  )
        log("Started instances", instance_list)
        
        instance_dict =  ec2_instance_getallstate()
        log("Instances running", instance_dict)

        ##### Launch Batch system by SSH  ####################################
        ipadress_list = [  x["ip_address"]  for k,x in instance_dict.items() ]
        for ipx in ipadress_list : 
          cmds = "bash /home/ubuntu/zs3drive/zbatch_cleanup.sh && which python && whoami &&  nohup bash /home/ubuntu/zs3drive/zbatch.sh ; "
          log(ipx, cmds)
          msg  = run_command_thru_ssh( ipx,  key_file,   cmds)
          #  cmdstr="nohup  /home/ubuntu/zbatch.sh  2>&1 | tee -a /home/ubuntu/zlog/zbatch_log.log")
          """
           Issues :
           1)   SSH command is time blocked....
           
           
           2) Issues with SH shell vs Bash Shell when doing SSH
           
           cmds = "bash /home/ubuntu/zbatch_cleanup.sh && which python && whoami &&  bash /home/ubuntu/zs3drive/zbatch.sh "
           
           
           ssh user@host "nohup command1 > /dev/null 2>&1 &; nohup command2; command3"
           ssh ubuntu@18.237.190.140 " /home/ubuntu/zbatch_cleanup.sh    && nohup  /home/ubuntu/zbatch.sh   "

           bash  nohup  bash /home/ubuntu/zbatch.sh
          
          """
                                
                                
          log("ssh",ipx, msg)
          sleep(5)
    
    
    
    ### Stop instance by rules ###############################################
    stop_instances = instance_stop_rule( args.task_folder)
    log("Instances to be stopped", stop_instances)
    if stop_instances:
      # ec2_instance_backup(stop_instances, folder_list=["/home/ubuntu/zlog/"])
      stop_instances_list = [v['id'] for v in stop_instances]
      ec2_instance_stop(stop_instances_list)
      log("Stopped instances", stop_instances_list)


    ### No Daemon mode  ######################################################
    if args.mode != "daemon":
      log("No Daemon mode","terminated daemon", os.getpid())
      break

    sleep(args.waitsec)






