# -*- coding: utf-8 -*-
"""
##### Daemon mode
batch_daemon_autoscale_cli.py --mode daemon --task_folder  zs3drive/tasks/  --log_file zlog/batchautoscale.log   
#### Test with reset task file, on S3 drive, no daemon mode
batch_daemon_autoscale_cli.py --task_folder  zs3drive/tasks/  --log_file zlog/batchautoscale.log   --reset_global_task_file 1
#### Test with reset of task files
batch_daemon_autoscale_cli.py --mode daemon --task_folder  zs3drive/tasks/  --log_file zlog/batchautoscale.log   --reset_global_task_file 1
#### Test with reset of task files and test p
batch_daemon_autoscale_cli.py  --mode daemon  --reset_global_task_file 1  --param_mode test   --param_file zs3drive/config_batch.toml  
#### Prod setup of task files
batch_daemon_autoscale_cli.py  --mode daemon  --reset_global_task_file 1 --param_file zs3drive/config_batch.toml  --param_mode prod
###########################################################################################
Daemon for auto-scale.
Only launch in master instance 
### S3 does NOT support folder rename, bash shell to replance rename
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
"""
import argparse
import copy
#################################################################################
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from time import sleep

import paramiko
from aapackage import util_log
from aapackage.batch import util_cpu
from aapackage.util_aws import aws_ec2_ssh
################################################################################
from aapackage.util_log import logger_setup

warnings.filterwarnings(action="ignore", module=".*paramiko.*")


############### Input  #########################################################
ISTEST = True  ### For test the code

cur_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(cur_path, "config.toml")


TASK_FOLDER_DEFAULT = os.path.dirname(os.path.realpath(__file__)) + "/ztest/tasks/"
# TASK_FOLDER_DEFAULT =  "/home/ubuntu/ztest/tasks/"

keypair = "aws_ec2_ajey"  # Remote Spot Instance
region = "us-west-2"  # Oregon West
default_instance_type = "t3.small"
amiId = "ami-090b189f0e2ffe4df"  # "ami-0d16a0996debff8d4"  #'ami-0491a657e7ed60af7'
spot_cfg_file = "/tmp/ec_spot_config"


### Record the running/done tasks on S3 DRIVE, Global File system  #############
global_task_file_default = "%s/zs3drive/ztest_global_task.json" % (
    os.environ["HOME"] if "HOME" in os.environ else "/home/ubuntu"
)


################################################################################
### Maintain infos on all instances  ###########################################
# global INSTANCE_DICT
INSTANCE_DICT = {
    "id": {"id": "", "cpu": 0, "ip_address": "", "ram": 0, "cpu_usage": 0, "ram_usage": 0}
}


################################################################################
logger = None


def log(*argv):
    logger.info(",".join([str(x) for x in argv]))


################################################################################
def os_folder_copy(from_folder_root, to_folder, isoverwrite=False, exclude_flag="ignore"):
    """
     ### Copy with criteria
  """
    task_list_added, task_list = [], []
    for f in os.listdir(from_folder_root):
        from_f = from_folder_root + f
        to_f = to_folder + f + "/"

        # Conditions of copy
        if os.path.isdir(from_f) and f not in {".git"} and exclude_flag not in f:
            if not os.path.exists(to_f) or isoverwrite:

                os.system("cp -r {f1} {f2}".format(f1=from_f, f2=to_f))
                print("Copy", from_f, to_f)

                if os.path.exists(to_f):
                    task_list_added.append(to_f)
                    task_list.append(from_f)
                else:
                    print("Error copy", from_f, to_f)
    return task_list, task_list_added


def task_get_from_github(
    repourl,
    reponame="tasks",
    branch="dev",
    to_task_folder="/home/ubuntu/zs3drive/tasks/",
    tmp_folder="/home/ubuntu/data/ztmp/",
):
    """
   Get tasks from github repo
   Retrieve tasks folder from Github repo and write on S3 drive for automatic processing.
   rm folder
   git clone  https://github.com/arita37/tasks.git
   git checkout branch
   for each subfolder :cp folder1  folder_s3
  """
    ### Git pull  ########################################################################
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    repo_folder = tmp_folder + "/" + reponame + "/"
    to_task_folder = to_task_folder + "/"

    msg = os.system("rm -rf " + repo_folder)
    cmds = " cd {a} && git clone {b}  {c}".format(a=tmp_folder, b=repourl, c=reponame)
    cmds += " && cd {a}  && git checkout {b}".format(a=reponame, b=branch)
    print(cmds)
    msg = os.system(cmds)
    # print(msg)

    ### Copy  ##########################################################################
    task_list, task_list_added = os_folder_copy(repo_folder, to_task_folder)

    ### Rename folder to ignore and commit #############################################
    for f in task_list:
        os.rename(f, f + "_ignore" if f[-1] != "/" else f[:-1] + "_ignore")

    cmds = " cd {a} && git add --all && git commit -m 'S3 copied '".format(a=repo_folder)
    cmds += " &&  git push --all --force "
    print(cmds)
    msg = os.system(cmds)

    return task_list, task_list_added


def task_put_to_github(
    repourl,
    reponame="tasks",
    branch="dev",
    from_taskout_folder="/home/ubuntu/zs3drive/tasks_out/",
    repo_folder="/home/ubuntu/data/github_tasks_out/",
):
    """
    Put results back to github
    git clone https://github.com/arita37/tasks_out.git  github_tasks_out
    
    git pull --all
    copy S3 to github_tasks_out
    git add --all, push all
  """
    if not os.path.exists(repo_folder):
        print("Git clone")
        cmds = " git clone {a} {b} ".format(a=repourl, b=repo_folder)
        cmds += " && git checkout {b}".format(b=branch)
        msg = os.system(cmds)

    cmds = " cd {a} && git pull --all ".format(a=repo_folder)
    cmds += " && git checkout {b}".format(b=branch)
    print("Git Pull results", cmds)
    msg = os.system(cmds)

    ### Copy with OVERWRITE
    task_list, task_list_added = os_folder_copy(from_taskout_folder, repo_folder, isoverwrite=True)

    ### Git push
    cmds = " cd {a} && git add --all  && git commit -m 'oo{b}'  ".format(
        a=repo_folder, b=",".join(task_list_added)
    )
    cmds += " && git push --all   --force "
    print("Git push task resut", cmds)
    msg = os.system(cmds)
    print(msg)
    return task_list, task_list_added


################################################################################
def task_get_list_valid_folder(folder, script_regex=r"main\.(sh|py)"):
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
            if re.match(script_regex, filename, re.I) and not re.match(
                r"^.*(_qstart|_qdone|_ignore)$", root_splits[-1], re.I
            ):
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
    task_all = {x for x in os.listdir(folder_main) if os.path.isdir("%s/%s" % (folder_main, x))}
    folders = list(task_all.difference(task_started))
    valid_folders = []
    for folder in folders:
        if task_isvalid_folder(folder_main, folder, folder_check):
            valid_folders.append(folder)

    print(valid_folders)
    return valid_folders


def task_isvalid_folder(folder_main, folder, folder_check):
    # Invalid cases
    if (
        os.path.isfile(os.path.join(folder_main, folder))
        or folder in folder_check
        or re.search(r"_qstart|_qdone|_ignore", folder, re.I)
    ):
        return False
    else:
        # Valid case
        return True


def task_getcount(folder_main):
    """ Number of tasks remaining to be scheduled for run """
    return len(task_get_list_valid_folder_new(folder_main))


def os_system(cmds, stdout_only=1):
    """
     Get print output from command line
  """
    import subprocess

    cmds = cmds.split(" ")
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.stdout.read(), p.stderr.read()

    if stdout_only:
        return out
    return out, err


def task_getcount_cpurequired(folder_main):
    """  
    ncpu_required defined in task_config.py
    
  
  """
    task_list = task_get_list_valid_folder_new(folder_main)
    ncpu_all = 0
    for f in task_list:
        cmds = "python  {a}/task_config.py --do ncpu_required   ".format(a=folder_main + "/" + f)
        msg = os_system(cmds)
        ncpu = parsefloat(msg, default=1.0)
        ncpu_all += ncpu
    return ncpu_all


##################################################################################
def ec2_get_spot_price(instance_type):
    """ Get the spot price for instance type in us-west-2"""
    value = 0.0
    if os.path.exists("./aws_spot_price.sh") and os.path.isfile("./aws_spot_price.sh"):
        cmdstr = "./aws_spot_price.sh %s | grep Price | awk '{print $2}'" % instance_type
        value = os.popen(cmdstr).read()
        value = value.replace("\n", "") if value else 0.10
        # parsefloat(value)
    return parsefloat(value)


def parsefloat(value, default=0.0):
    """ Parse the float value. """
    try:
        return float(value)
    except:
        return default


################################################################################
def instance_get_ncpu(instances_dict):
    """ Total cpu count for the launched instances. """
    ss = 0
    if instances_dict:
        for _, x in instances_dict.items():
            ss += x["cpu"]
    return ss


################################################################################
def ec2_instance_getallstate():
    """
      use to update the global INSTANCE_DICT
          "id" :  instance_type,
          ip_address, cpu, ram, cpu_usage, ram_usage
  """
    val = {}
    spot_list = ec2_spot_instance_list()
    spot_instances = []
    for spot_instance in spot_list["SpotInstanceRequests"]:
        if re.match(spot_instance["State"], "active", re.I) and "InstanceId" in spot_instance:
            spot_instances.append(spot_instance["InstanceId"])
    # print(spot_instances)

    for spot in spot_instances:
        cmdargs = ["aws", "ec2", "describe-instances", "--instance-id", spot]
        cmd = " ".join(cmdargs)
        value = os.popen(cmd).read()
        inst = json.loads(value)
        ncpu = 0
        ipaddr = None
        instance_type = default_instance_type
        if inst and "Reservations" in inst and inst["Reservations"]:
            reserves = inst["Reservations"][0]
            if "Instances" in reserves and reserves["Instances"]:
                instance = reserves["Instances"][0]

                if "CpuOptions" in instance and "CoreCount" in instance["CpuOptions"]:
                    ncpu = instance["CpuOptions"]["CoreCount"]

                if "PublicIpAddress" in instance and instance["PublicIpAddress"]:
                    ipaddr = instance["PublicIpAddress"]

                instance_type = instance["InstanceType"]

        if ipaddr:
            cpuusage, usageram, totalram = ec2_instance_usage(spot, ipaddr)
            # print(cpuusage, usageram, totalram)
            val[spot] = {
                "id": spot,
                "instance_type": instance_type,
                "cpu": ncpu,
                "ip_address": ipaddr,
                "ram": totalram,
                "cpu_usage": cpuusage,
                "ram_usage": usageram,
            }
    # print(val)
    return val


################################################################################
def ec2_keypair_get():
    identity = "%s/.ssh/%s" % (
        os.environ["HOME"] if "HOME" in os.environ else "/home/ubuntu",
        keypair,
    )
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
        cpuusage = ssh_cmdrun(ipadress, identity, cmdstr)
        cpuusage = 100.0 if not cpuusage else float(cpuusage)

        cmdstr = "free | grep Mem | awk '{print $3/$2 * 100.0, $2}'"
        # ram = ssh.command(cmdstr)
        ramusage = ssh_cmdrun(ipadress, identity, cmdstr)

        if not ramusage:
            totalram = 0
            usageram = 100.0
        else:
            vals = ramusage.split()
            usageram = float(vals[0]) if vals and vals[0] else 100.0
            totalram = int(vals[1]) if vals and vals[1] else 0

    return cpuusage, usageram, totalram


################################################################################
def ec2_config_build_template(instance_type):
    """ Build the spot json config into a json file. """
    spot_config = {
        "ImageId": amiId,
        "KeyName": keypair,
        "SecurityGroupIds": ["sg-4b1d6631", "sg-42e59e38"],
        "InstanceType": instance_type if instance_type else default_instance_type,
        "IamInstanceProfile": {"Arn": "arn:aws:iam::013584577149:instance-profile/ecsInstanceRole"},
        "BlockDeviceMappings": [
            {"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 60}}
        ],
    }
    with open(spot_cfg_file, "w") as spot_file:
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
    ec2_config_build_template(instance_type)
    cmdargs = [
        "aws",
        "ec2",
        "request-spot-instances",
        "--region",
        region,
        "--spot-price",
        str(spot_price),
        "--instance-count",
        "1",
        " --type",
        "one-time",
        "--launch-specification",
        "file://%s" % spot_cfg_file,
    ]
    print(cmdargs)
    cmd = " ".join(cmdargs)
    msg = os.system(cmd)
    sleep(waitsec)  # It may not be fulfilled in 50 secs.
    ll = ec2_spot_instance_list()
    return ll["SpotInstanceRequests"] if "SpotInstanceRequests" in ll else []


def ec2_spot_instance_list():
    """ Get the list of current spot instances. """
    cmdargs = ["aws", "ec2", "describe-spot-instance-requests"]
    cmd = " ".join(cmdargs)
    value = os.popen(cmd).read()
    try:
        instance_list = json.loads(value)
    except:
        instance_list = {"SpotInstanceRequests": []}
    return instance_list


################################################################################
def ec2_instance_stop(instance_list):
    """ Stop the spot instances ainstances u stop any other instance, this should work"""
    instances = instance_list
    if instances:
        if isinstance(instance_list, list):
            instances = ",".join(instance_list)
        cmdargs = ["aws", "ec2", "terminate-instances", "--instance-ids", instances]
        cmd = " ".join(cmdargs)
        os.system(cmd)
        return instances.split(",")


################################################################################
def ec2_instance_backup(
    instances_list, folder_list=["zlog/"], folder_backup="/home/ubuntu/zs3drive/backup/"
):
    """
      Zip some local folders
      Tansfer data from local to /zs3drive/backup/AMIname_YYYYMMDDss/
    tar -czvf directorios.tar.gz folder
    
    """
    from datetime import datetime

    now = datetime.today().strftime("%Y%m%d")
    for idx in instances_list:

        target_folder = folder_backup + "/a" + now + "_" + idx

        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        for f in folder_list:
            # fname = f.split("/")[-1]
            cmds = "cp -r {a} {b}".format(a=f, b=target_folder)
            msg = os.system(cmds)
            print(cmds, msg)

        """
      # ssh = aws_ec2_ssh( inst["ip_address"], key_file)
      target_folder = folder_backup +  "/" + inst["id"] +  "_" + now
      cmdstr = "mkdir %s" % target_folder
      print(cmds)
      
      msg = ssh_cmdrun( inst["ip_address"],  key_file,   cmds, True)
      print(msg)      
      for t in folder_list :
        cmds = "tar -czvf  %s/%s.tar.gz %s" % (target_folder,
                                               t.replace('/', ''), t)
        print(cmds)
        # ssh.cmd(cmdstr)
        msg = ssh_cmdrun( inst["ip_address"],  key_file,   cmds, True)
      """


################################################################################
def instance_start_rule(task_folder):
    """ Start spot instance if more than 10 tasks or less than 10 CPUs 
      return instance type, spotprice
  """
    global INSTANCE_DICT
    # ntask = task_getcount(task_folder)
    ntask = task_getcount_cpurequired(task_folder)
    ncpu = instance_get_ncpu(INSTANCE_DICT)
    log("Start Rule", "Ntask, ncpu", ntask, ncpu)

    if ntask == 0 and not ISTEST:
        return None

    # hard coded values here
    if ntask > 20 and ncpu < 5:
        # spotprice = max(0.05, ec2_get_spot_price('t3.medium')* 1.30)
        spotprice = 0.05
        return {"type": "t3.medium", "spotprice": spotprice}

    if ntask > 15 and ncpu < 3:
        # spotprice = max(0.05, ec2_get_spot_price('t3.medium')* 1.30)
        # 8 CPU, 0.10 / hour
        spotprice = 0.15
        return {"type": "t3.2xlarge", "spotprice": spotprice}

    ##### Minimal instance  ###################################################
    if ntask > 0 and ncpu == 0:
        # spotprice = max(0.05, ec2_get_spot_price('t3.medium')* 1.30)
        # 2 CPU / 0.02 / hour
        spotprice = 0.05
        return {"type": "t3.medium", "spotprice": spotprice}

    return None


def instance_stop_rule(task_folder):
    """IF spot instance usage is ZERO CPU%  and RAM is low --> close instances.
  
  """
    global INSTANCE_DICT
    # ntask              = task_getcount(task_folder)
    ntask = task_getcount_cpurequired(task_folder)
    INSTANCE_DICT_prev = copy.deepcopy(INSTANCE_DICT)
    INSTANCE_DICT = ec2_instance_getallstate()
    log("Stop rules", "ntask", ntask, INSTANCE_DICT)

    if ntask == 0 and INSTANCE_DICT:
        # Idle Instances
        instance_list = [
            x for _, x in INSTANCE_DICT.items() if x["cpu_usage"] < 10.0 and x["ram_usage"] < 9.0
        ]

        return instance_list
    else:
        return None

    """    
  instance_list = []
  
  if ntask == 0 and  INSTANCE_DICT :
      # Idle Instances
      for idx, x in INSTANCE_DICT.items() :
         if x["cpu_usage"] < 10.0   and x["ram_usage"] < 8.0 :
            instance_list.append(x)  
      return instance_list
  else :
      return None
  """


def ssh_cmdrun(hostname, key_file, cmdstr, remove_newline=True, isblocking=True):
    """ Make an ssh connection using paramiko and  run the command
   http://sebastiandahlgren.se/2012/10/11/using-paramiko-to-send-ssh-commands/
   https://gist.github.com/kdheepak/c18f030494fea16ffd92d95c93a6d40d
   https://github.com/paramiko/paramiko/issues/501
   https://unix.stackexchange.com/questions/30400/execute-remote-commands-completely-detaching-from-the-ssh-connection
   
  """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, key_filename=key_file, timeout=5)
        stdin, stdout, stderr = ssh.exec_command(cmdstr)  # No Blocking  , get_pty=False

        """
    if not isblocking :
       # Buggy code, use Screen instead
       sleep(10) # To let run the script
       ssh.close()
       return None
    """

        #### Can be Blocking for long running process  screen -d -m YOURBASH
        data = stdout.readlines()  # Blocking code
        value = "".join(data).replace("\n", "") if remove_newline else "".join(data)

        err_msg = stderr.readlines()
        if len(err_msg) > 0:
            print(err_msg)

        ssh.close()
        return value

    except Exception as e:
        print("Error Paramiko", e)
        return None


def ssh_put(hostname, key_file, remote_file, msg=None, filename=None):
    """ Make an ssh connection using paramiko and  run the command
     http://sebastiandahlgren.se/2012/10/11/using-paramiko-to-send-ssh-commands/
     https://gist.github.com/kdheepak/c18f030494fea16ffd92d95c93a6d40d
 
     https://github.com/paramiko/paramiko/issues/501
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, key_filename=key_file, timeout=5)
    # stdin, stdout, stderr = ssh.exec_command(cmdstr, get_pty=False) #No Blocking

    if filename is not None:
        msg = open(filename, mode="r").readlines()

    ftp = ssh.open_sftp()
    file = ftp.file(remote_file, "a", -1)
    file.write(msg)
    file.flush()
    ftp.close()
    ssh.close()


def ec2_instance_initialize_ssh(args):
    """
          Many issues with S3 and ssh, Very sensitive code...
          1) Cannot run bash shell from S3 drive folder
          2) Screen uses SH shell, not bash ---> Need to add .bashrc,python path in main.sh script
             see task_template/
        """
    ##### Launch Batch system by No Blocking SSH  ####################################
    for k, x in INSTANCE_DICT.items():
        ipx = x["ip_address"]
        instance_id = x["id"]
        # msg= """#!/bin/bash
        #     bash /home/ubuntu/zs3drive/zbatch_cleanup.sh && which python && whoami &&  nohup bash /home/ubuntu/zs3drive/zbatch.sh
        #     """
        # ssh_put(ipx , key_file, "/home/ubuntu/zbatch_ssh.sh", msg)

        #### issues with access
        cmds = " cp /home/ubuntu/zs3drive/zbatch_cleanup.sh  /home/ubuntu/zbatch_cleanup.sh   "
        cmds += " && cp /home/ubuntu/zs3drive/zbatch.sh  /home/ubuntu/zbatch.sh   "
        cmds += (
            " && chmod 777 /home/ubuntu/zbatch_cleanup.sh && chmod 777 /home/ubuntu/zbatch.sh   "
        )
        cmds += " && echo  ' copied'   "
        # msg  = ssh_cmdrun( ipx,  key_file,   cmds, isblocking=True)
        # log(ipx, "ssh copy script file to Local", msg)

        cmds += " chmod 777 /home/ubuntu/zbatch_test.sh && chmod 777 /home/ubuntu/zbatch.sh   "
        cmds += " && bash /home/ubuntu/zbatch_cleanup.sh    "
        cmds += " && which python && echo  ',' && pwd "
        msg = ssh_cmdrun(ipx, key_file, cmds, isblocking=True)
        log(ipx, "ssh zbatch_cleanup", msg)

        #### MAJOR BUG : CANNOT USE bash script on S3 Folder ,due to Permission ISSUES on S#
        #### Neeed to add anaconda into the path
        if "test" in args.param_mode:
            cmds = " screen -d -m bash /home/ubuntu/zbatch_test.sh && sleep 5  && screen -ls "
        else:
            cmds = " screen -d -m bash /home/ubuntu/zbatch.sh && sleep 5  && screen -ls "

        # cmds += " screen -d -m bash /home/ubuntu/zs3drive/zbatch.sh && screen -ls "

        log(ipx, "no blocking mode ssh", cmds)
        msg = ssh_cmdrun(ipx, key_file, cmds, isblocking=True)
        log(ipx, "ssh zbatch.sh", msg)
        if "Socket" not in str(msg):  # Screen is not launched....
            log(ipx, "MAJOR ISSUE, daemon_launcher NOT launched")
            sleep(10)
            log(ipx, "Terminating", instance_id, ipx)
            ec2_instance_stop(instance_list=[instance_id])


def task_globalfile_reset(global_task_file=None):
    with open(global_task_file, "w") as f:
        json.dump({}, f)


################################################################################
def load_arguments():
    """
     Load CLI input, load config.toml , overwrite config.toml by CLI Input
  """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, "config.toml")

    p = argparse.ArgumentParser()
    p.add_argument("--param_file", default=config_file, help="Params File")
    p.add_argument("--param_mode", default="test", help=" test/ prod /uat")

    p.add_argument("--mode", help="daemon/ .")  # default="nodaemon",

    p.add_argument("--log_file", help=".")  # default="batchdaemon_autoscale.log",

    p.add_argument(
        "--global_task_file", help="global task file"
    )  #  default=global_task_file_default,
    p.add_argument("--task_folder", help="path to task folder.")  # default=TASK_FOLDER_DEFAULT,
    p.add_argument("--reset_global_task_file", help="global task file Reset File")

    p.add_argument(
        "--task_repourl", help="repo for task"
    )  # default="https://github.com/arita37/tasks.git"
    p.add_argument("--task_reponame", help="repo for task")  # default="tasks",
    p.add_argument("--task_repobranch", help="repo for task")  #  default="dev",

    p.add_argument("--ami", help="AMI used for spot")  #  default=amiId,
    p.add_argument("--instance", help="Type of soot instance")  # default=default_instance_type,
    p.add_argument("--spotprice", type=float, help="Actual price offered by us.")
    p.add_argument("--waitsec", type=int, help="wait sec")
    p.add_argument("--max_instance", type=int, help="")
    p.add_argument("--max_cpu", type=int, help="")
    args = p.parse_args()

    ##### Load file params as dict namespace #########################
    import toml

    class to_namespace(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    print(args.param_file)
    pars = toml.load(args.param_file)
    # print(args.param_file, pars)
    pars = pars[args.param_mode]  # test / prod
    print(args.param_file, pars)

    ### Overwrite params by CLI input and merge with toml file
    for key, x in vars(args).items():
        if x is not None:  # only values NOT set by CLI
            pars[key] = x

    # print(pars)
    pars = to_namespace(pars)  #  like object/namespace pars.instance
    return pars


###################################################################################
if __name__ == "__main__":
    ### Variable initialization #####################################################
    args = load_arguments()

    logger = logger_setup(
        __name__, log_file=args.log_file, formatter=util_log.FORMATTER_4, isrotate=True
    )
    # print("args input", args)
    key_file = ec2_keypair_get()

    global_task_file = args.global_task_file
    if args.reset_global_task_file:
        task_globalfile_reset(global_task_file)

    log("Daemon", "start: ", os.getpid(), global_task_file)
    ii = 0
    while True:
        log("Daemon", "tasks folder: ", args.task_folder)

        ### Retrieve tasks from github ##############################################
        if ii % 5 == 0:
            task_new, task_added = task_get_from_github(
                repourl=args.task_repourl,
                reponame=args.task_reponame,
                branch=args.task_repobranch,
                to_task_folder=args.task_s3_folder,  # r"/home/ubuntu/zs3drive/tasks/",
                tmp_folder=args.task_local_folder,
            )  # r"/home/ubuntu/data/ztmp_github/")
            log("task", "new from github", task_added)

        # Keep Global state of running instances
        INSTANCE_DICT = ec2_instance_getallstate()

        ### Start instance by rules ###############################################
        start_instance = instance_start_rule(args.task_folder)
        log("Instances to start", start_instance)
        if start_instance:
            # When instance start, batchdaemon will start and picks up task in  COMMON DRIVE /zs3drive/
            instance_list = ec2_spot_start(start_instance["type"], start_instance["spotprice"])
            log("Instances started", instance_list)

            INSTANCE_DICT = ec2_instance_getallstate()
            log("Instances running", INSTANCE_DICT)

            ##### Launch Batch system by No Blocking SSH  #########################
            ec2_instance_initialize_ssh(args)
            sleep(10)

        ### Stop instance by rules ################################################
        stop_instances = instance_stop_rule(args.task_folder)
        log("Instances to be stopped", stop_instances)
        if stop_instances:
            stop_instances_list = [v["id"] for v in stop_instances]

            ec2_instance_backup(
                stop_instances_list,
                folder_list=args.folder_to_backup,  # ["/home/ubuntu/zlog/", "/home/ubuntu/tasks_out/" ],
                folder_backup=args.backup_s3_folder,
            )  # "/home/ubuntu/zs3drive/backup/"

            ec2_instance_stop(stop_instances_list)
            log("Stopped instances", stop_instances_list)

        ### Upload results to github ##############################################
        ii = ii + 1
        if ii % 10 == 0:  # 10 mins Freq
            task_new, task_added = task_put_to_github(
                repourl=args.taskout_repourl,  # "https://github.com/arita37/tasks_out.git"
                reponame=args.taskout_reponame,
                branch=args.taskout_repobranch,  # "tasks_out", branch="dev",
                from_taskout_folder=args.taskout_s3_folder,  # "/home/ubuntu/zs3drive/tasks_out/"
                repo_folder=args.taskout_local_folder,
            )  # "/home/ubuntu/data/github_tasks_out/"
            log("task", "Add results to github", task_added)

        ### No Daemon mode  ######################################################
        if args.mode != "daemon":
            log("Daemon", "No Daemon mode", "terminated daemon")
            break

        sleep(args.waitsec)


"""
if not ISTEST :
  ### Global Shared Drive
  TASK_S3_FOLDER    = "/home/ubuntu/zs3drive/tasks/"
  BACKUP_S3_FOLDER  = "/home/ubuntu/zs3drive/backup/"
  TASKOUT_S3_FOLDER = "/home/ubuntu/zs3drive/tasks_out/"
  ### Local to each instance
  TASKOUT_REPOURL      = "https://github.com/arita37/tasks_out.git"
  TASKOUT_LOCAL_FOLDER = "/home/ubuntu/data/github_tasks_out/"
  TASK_REPOURL      = "https://github.com/arita37/tasks.git"
  TASK_LOCAL_FOLDER = "/home/ubuntu/data/github_tasks/"
  FOLDER_TO_BACKUP  = ["/home/ubuntu/zlog/", "/home/ubuntu/tasks_out/" ]
  ### Record the running/done tasks on S3 DRIVE, Global File system  #############
  global_task_file = "%s/zs3drive/global_task.json" % (os.environ['HOME'] 
                     if 'HOME' in os.environ else '/home/ubuntu')
else  :
  ### Global Shared Drive
  TASK_S3_FOLDER    = "/home/ubuntu/zs3drive/ztest_tasks/"
  BACKUP_S3_FOLDER  = "/home/ubuntu/zs3drive/ztest_backup/"
  TASKOUT_S3_FOLDER = "/home/ubuntu/zs3drive/ztest_tasks_out/"
  ### Local to each instance
  TASKOUT_REPOURL      = "https://github.com/arita37/tasks_out.git"
  TASKOUT_LOCAL_FOLDER = "/home/ubuntu/data/ztest_github_tasks_out/"
  TASK_REPOURL      = "https://github.com/arita37/tasks.git"
  TASK_LOCAL_FOLDER = "/home/ubuntu/data/ztest_github_tasks/"
  FOLDER_TO_BACKUP  = ["/home/ubuntu/zlog/", "/home/ubuntu/tasks_out/" ]
"""


"""
Problem o blocking
          cmds = "chmod 777  /home/ubuntu/zbatch_ssh.sh &&  screen -d -m  bash /home/ubuntu/zbatch_ssh.sh"
          cmds = "bash /home/ubuntu/zs3drive/zbatch_cleanup.sh && which python && whoami &&  nohup bash /home/ubuntu/zs3drive/zbatch.sh </dev/null >/dev/null 2>&1 & "   
            f.write(cmds)     
            
            
Found this on google groups: Starting a daemon with ssh - comp.unix.admin | Google Groups
ssh server 'program </dev/null >/dev/null 2>&1 &' 
that redirects the stdin to /dev/null, the stdout to /dev/null, and the stderr to stdout
This worked for me so that the remote execution kicked off the daemon and didn't wait around for output.
Will
           1)   SSH command is time blocked....
           
           https://github.com/paramiko/paramiko/issues/501
           
           set
get_pty = False
and use
nohup /tmp/b.sh >> /tmp/a.log 2>>/tmp/a.log &
           whic
       
           2) Issues with SH shell vs Bash Shell when doing SSH
               need to load bashrc manually
           #  cmdstr="nohup  /home/ubuntu/zbatch.sh  2>&1 | tee -a /home/ubuntu/zlog/zbatch_log.log")
           cmds = "bash /home/ubuntu/zbatch_cleanup.sh && which python && whoami &&  bash /home/ubuntu/zs3drive/zbatch.sh "
           ssh user@host "nohup command1 > /dev/null 2>&1 &; nohup command2; command3"
           ssh ubuntu@18.237.190.140 " /home/ubuntu/zbatch_cleanup.sh    && nohup  /home/ubuntu/zbatch.sh   "
          
          
          
  # Send the command (non-blocking)
stdin, stdout, stderr = ssh.exec_command("my_long_command --arg 1 --arg 2")
# Wait for the command to terminate
while not stdout.channel.exit_status_ready():
    # Only print data if there is data to read in the channel
    if stdout.channel.recv_ready():
        rl, wl, xl = select.select([stdout.channel], [], [], 0.0)
        if len(rl) > 0:
            # Print data from stdout
            print stdout.channel.recv(1024),
"""
