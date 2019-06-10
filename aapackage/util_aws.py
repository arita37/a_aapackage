# -*- coding: utf-8 -*-
"""
---------AWS utilities--------------------------------------------------------
Usage:
AWS is class that defines all the configuration like access key or secret or pem key,
which can be used  both by ec2, s3, rds connections. At this point, it has been used for
ec2 and s3 connections.
AWS object can be populated from a file containing a json object
Eg: Use pem file to make ssh connection
    import socket, paramiko
    from util_aws import AWS
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(hostname, 22)
    transport = paramiko.Transport(self.sock)
    transport.start_client()
    pemfile = AWS().v['AWS_KEYPEM'] # Get the private key pem file
    privatekey = paramiko.RSAKey.from_private_key_file(pemfile)
    transport.auth_publickey(username, privatekey)

Eg: Get the keypair stored in the AWS for running a command using SSH
    identity = AWS().ec2_keypair_get()  # Get the key pair
    cmdstr = "df -k | grep swap"
    swapspace = ssh_cmdrun(ipadress, identity, cmdstr)

Eg: Get the access and secret key for accessing AWS
    access, secret = AWS().aws_accesskey_get()
    conn = boto.connect_s3(access, secret)
    bucket = conn.get_bucket(bucket_name)

Eg: Create a conn for us-east-2
    east2-conn = AWS().aws_conn_create(region='us-east-2')

Eg: Create windows conn for us-east-1
    east1-win-conn = AWS().aws_conn_create_windows(aws_region='us-east-1')

Utility methods:
ssh_cmdrun(): Runs a specific command using ssh library called paramiko and pem key file.
  ipaddress = '12.30.24.21'
  identity = AWS().ec2_keypair_get()
  cmdstr = "ls -lrt"
  reverselisting = ssh_cmdrun(ipadress, identity, cmdstr)

For a instance type, fetch ram, cpu usage as an array for spot requests only
  ec2_instance_getallstate_cli('t2.small')

Each instance usage details
  spot_requestid = '<spot request id>'
  ipaddr = '12.30.15.25'
  cpuusage, usageram, totalram = ec2_instance_usage(spot_requestid, ipaddr)

Build a template for a spot request and store in default location(/tmp/ec_spot_config)
  ec2_config_build_template_cli('t2.medium')


Request spot instance.
  instance = 't3.small'
  spot_price = 0.59
  region = 'us-east-1'
  ec2_spot_start_cli(instance, spot_price, region)

Current list of spot instances.
  spot_list = ec2_spot_instance_list()


Stop the spot instance based on the instanceid
  spot_request_id = '<Req Id>'
  ec2_instance_stop(spot_request_id)

For a given instance type, run a shell script to make a http call
and grep the price for the same.
  instance_type = 't3.medium'
  ec2_get_spot_price(instance_type)




TODO :
  All in one file util_aws.py  : Top priority
 - Remove Dependance from util.py
 
 - Create/check  /USER/.aws/ folder to put all the HARD CODE values. 
    .ssh : for SSH keys.

 - AWS: the default values should be in a default file, not in code
        This default json file should be in the same directory as this file.

 - boto.config.get in lines 148 and 149 are not working, they need to be fixed.
 - Have imports at the top unless dependencies issues are to be resolved.
 
 Short name is better - AWS.v is not good descriptive name for a instance variable, could be AWS.constant_values
    or AWS.global_values.

 - Lines 206-212, optimize as EC2Connection(access, key, region=r), if it fails then check
    if it is a valid region. Anyway as of now 16 valid regions are there and they can be
    statically checked, why fetch them?

 - Dont see a difference between a  windows and a non-windows EC2 connection.

 - ssh_cmdrun, ssh_put  methods to be moved to a separate file, can be used by other files,
    which do not require other AWS functionality. Platform check for Windows machine and
    return a False. Also adapt it to run a list of commands as it saves making connection
    multiple times. In lines 366-388, 2 commands are run on the same host and the connections
    are made twice.
 - os.system does not return output, os.popen does return.  For all aws ec2 commands, it is
    json output, so always good to capture it.
 - aws_ec2_get_instanceid, this will need pagination support, an account may have more than
    one page of instances.
 - aws_ec2_allocate_elastic_ip and aws_ec2_allocate_eip are same code set, need a merge. Multiple
    times defined (Lines 535 and 746)



NG - json.load(<filename>) is notorius to fail, so have a json_utils.py and have a
        try-catch block with fallback of returning a empty dict.  

NG - Have all these constants in a separate file constants.py and import
   only required constants like
   from constants import AWS_ACCESS_LOCAL
   This way we bring structure to the code and anybody can update and maintain it.

NG -  Keep CLI and boto version separate.separate
      aws_ec2_spot_start and aws_ec2_spot_start_cli are same, one uses boto and the other aws ec2.
     the code should switch between these two calls based on configuration ( global or individual calls)
 -
"""

from __future__ import division, print_function

import csv
###############################################################################
import json
import os
import re
import sys
from time import sleep

import boto
from boto import ec2
from boto.ec2.blockdevicemapping import BlockDeviceMapping, EBSBlockDeviceType
from future import standard_library

import paramiko
from aapackage import util  # want to remove after refactor
standard_library.install_aliases()


###############################################################################
class AWS:
    """
    All the globals for AWS utility functionalities.
    """

    def __init__(self, name=None, keypair=None, keypem=None):
        """Nothing to be constructed """
        self.v = {
            "AWS_CONFIG_FOLDER" : "/.aws/",
            "AWS_ACCESS_LOCAL": 'D:/_devs/keypair/aws_access.py',
            "AWS_KEYPEM": keypem if keypem else  "D:/_devs/keypair/oregon/aws_ec2_oregon.pem",
            "AWS_KEYPAIR": keypair if keypair else "aws_ec2_oregon",
            "AWS_REGION": "us-west-2",
            # "APNORTHEAST2": 'ap-northeast-2',   ## Not good.. HARD coded  APNORTHEAST2

            "DEFAULT_INSTANCE_TYPE": "t3.small",
            "EC2CWD": '/home/ubuntu/',
            "EC2_CONN": None,
            "EC2_FILTERS": ('id', 'ip_address'),
            "EC2_ATTRIBUTES": (
                "id", "instance_type", "state", "public_dns_name", "private_dns_name",
                "state_code", "previous_state", "previous_state_code", "key_name",
                "launch_time", "image_id", "placement", "placement_group",
                "placement_tenancy", "kernel", "ramdisk", "architecture", "hypervisor",
                "virtualization_type", "product_codes", "ami_launch_index", "monitored",
                "monitoring_state", "spot_instance_request_id", "subnet_id", "vpc_id",
                "private_ip_address", "ip_address", "platform", "root_device_name",
                "root_device_type", "state_reason", "interfaces", "ebs_optimized",
                "instance_profile"
            ),
            "SPOT_CFG_FILE": ".aws/ec_spot_config"
        }
        
        if ".json" in name:
            dd = json.load(name)
        else:
            dd = json.load( os["HOME"] + "/.aws/" + name )
        if dd:
            self.v.update(dd)
        self.v = dict2(self.v)
        self.conn = None

    def aws_accesskey_get(self):
        """Return a tuple of AWS credentials (access key id and secret access key)"""
        # Try from Boto Config
        access = boto.config.get('Credentials', 'aws_access_key_id')
        key = boto.config.get('Credentials', 'aws_secret_access_key')

        if access and key:
            print(access, key)
            return access, key

        # Finally try the manual Config
        with open(self.v['AWS_ACCESS_LOCAL']) as aws_file:
            dd = json.loads(aws_file.read())
        if 'AWS_ACCESS_KEY_ID' in dd:
            access = dd['AWS_ACCESS_KEY_ID']
        if 'AWS_SECRET_ACCESS_KEY' in dd:
            key = dd['AWS_SECRET_ACCESS_KEY']
        return access, key

    def ec2_keypair_get(self, keypair=""):
        sshdir = "%s/.ssh" % os.environ['HOME'] if 'HOME' in os.environ else '/home/ubuntu'
        keypair = keypair if keypair else self.v['AWS_KEYPEM']
        identity = '%s/%s' % (sshdir, keypair)
        return identity

    def get_keypair(self):
        """Get the current keypair used"""
        return self.v['AWS_KEYPEM'], self.v['AWS_KEYPAIR']

    def set_keypair(self, keypairname, keypairlocation):
        """Set the keypair to be used."""
        if keypairname and keypairlocation:
            self.v['AWS_KEYPAIR'] = keypairname
            self.v['AWS_KEYPEM'] = keypairlocation

    @classmethod
    def set_attribute(cls, key, value):
        """Add or update attribute to the class, maybe protect with a lock."""
        setattr(cls, key, value)

    def get_ec2_conn(self):
        """Return the current EC2 connection."""
        return self.conn

    def aws_conn_create(self, region='', access='', key=''):
        """ EC2 connection to AP NORTH EAST2 """
        from boto.ec2.connection import EC2Connection
        if not region:
            region = self.v['APNORTHEAST2']
        if not access or not key:
            access, key = self.aws_accesskey_get()
        if not access or not key:
            return None
        # We could have used connection from any region.
        conn = EC2Connection(access, key)
        regions = aws_conn_getallregions(conn)
        for r in regions:
            if r.name == region:
                self.conn = EC2Connection(access, key, region=r)
                return self.conn
        print('Region not Found')
        return None

    def aws_conn_create_windows(self, aws_region=''):
        """ Return ec2_conn for windows system, otherwise none."""
        if not aws_region:
            aws_region = self.v['AWS_REGION']
        ec2_conn = self.conn
        if sys.platform.find('win') > -1 and not ec2_conn:
            dd = {}
            with open(self.v['AWS_ACCESS_LOCAL']) as aws_file:
                dd = json.loads(aws_file.read())
            if 'AWS_ACCESS_KEY_ID' in dd:
                access = dd['AWS_ACCESS_KEY_ID']
            if 'AWS_SECRET_ACCESS_KEY' in dd:
                key = dd['AWS_SECRET_ACCESS_KEY']
            if access and key:
                ec2_conn = boto.ec2.connect_to_region(aws_region, aws_access_key_id=access,
                                                      aws_secret_access_key=key)
                print(ec2_conn)
                # We should store this in AWS
                self.conn = ec2_conn
        return self.conn


###############StandAlone access ###################################################################
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
    stdin, stdout, stderr = ssh.exec_command(cmdstr) #No Blocking  , get_pty=False
    
    """
    if not isblocking :
       # Buggy code, use Screen instead
       sleep(10) # To let run the script
       ssh.close()
       return None
    """
    
    #### Can be Blocking for long running process  screen -d -m YOURBASH
    data  = stdout.readlines()  #Blocking code
    value = ''.join(data).replace('\n', '') if remove_newline else ''.join(data)
    
    err_msg = stderr.readlines() 
    if len(err_msg) > 0 :
      print(err_msg  )
    
    ssh.close()
    return value
  except Exception as e :
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
    #stdin, stdout, stderr = ssh.exec_command(cmdstr, get_pty=False) #No Blocking 
    
    if filename is not None :
      msg = open(filename,mode="r").readlines()
    
    ftp = ssh.open_sftp()
    file=ftp.file(remote_file, "a", -1)
    file.write(msg)
    file.flush()
    ftp.close()
    ssh.close()


def ec2_instance_getallstate_cli(default_instance_type="t3.medium"):
  """
  use to update the global INSTANCE_DICT
  "id" :  instance_type, ip_address, cpu, ram, cpu_usage, ram_usage
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
    cmd     = ' '.join(cmdargs)
    value   = os.popen(cmd).read()
    inst    = json.loads(value)
    ncpu    = 0
    ipaddr  = None
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
def ec2_instance_usage(instance_id=None, ipadress=None):
  """
  https://stackoverflow.com/questions/20693089/get-cpu-usage-via-ssh
  https://haloseeker.com/5-commands-to-check-memory-usage-on-linux-via-ssh/
  """
  cpuusage, ramusage, totalram = [None] * 3
  totalram = None
  if instance_id and ipadress:
    identity = AWS().ec2_keypair_get()
    cmdstr = "top -b -n 10 -d.2 | grep 'Cpu' | awk 'BEGIN{val=0.0}{ if( $2 > val ) val = $2} END{print(val)}'"
    cpuusage = ssh_cmdrun(ipadress, identity, cmdstr)
    cpuusage = 100.0  if not cpuusage else float(cpuusage)
    cmdstr = "free | grep Mem | awk '{print $3/$2 * 100.0, $2}'"
    ramusage = ssh_cmdrun(ipadress, identity, cmdstr)
    if not ramusage:
        totalram = 0
        usageram = 100.0
    else:
        vals = ramusage.split()
        usageram = float(vals[0]) if vals and vals[0] else 100.0
        totalram = int(vals[1]) if vals and vals[1] else 0
  return cpuusage, usageram, totalram


def ec2_config_build_template_cli(instance_type, amiId=None, keypair=None, 
         default_instance_type=None, spot_cfg_file=None):
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
  if not spot_cfg_file:
      spot_cfg_file = AWS().v['SPOT_CFG_FILE']
  with open(spot_cfg_file, 'w') as spot_file:
    spot_file.write(json.dumps(spot_config))


def ec2_spot_start_cli(instance_type, spot_price, region=None, waitsec=100):
  """
  Request a spot instance based on the price for the instance type
  # Need a check if this request has been successful.
  
  100 sec to be provisionned and started.
  """
  aws = AWS()
  if not instance_type:
      instance_type = aws.v['DEFAULT_INSTANCE_TYPE']
  if not region:
      region = aws.v['AWS_REGION']
  ec2_config_build_template_cli(instance_type)
  cmdargs = [
    'aws', 'ec2', 'request-spot-instances',
    '--region', region,
    '--spot-price', str(spot_price),
    '--instance-count', "1",
    ' --type', 'one-time',
    '--launch-specification', 'file://%s' % aws.v['SPOT_CFG_FILE']
  ]
  print(cmdargs)
  cmd = ' '.join(cmdargs)
  os.system(cmd)
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


def ec2_get_spot_price(instance_type):
  """ Get the spot price for instance type in us-west-2"""
  value = 0.0
  if os.path.exists('./aws_spot_price.sh') and os.path.isfile('./aws_spot_price.sh'):
    cmdstr = "./aws_spot_price.sh %s | grep Price | awk '{print $2}'" % instance_type
    value = os.popen(cmdstr).read()
    value = value.replace('\n', '') if value else 0.10
  return tofloat(value)


def aws_conn_getallregions(conn=None):
    """ Get the regions for the connection passed. """
    if conn:
        return conn.get_all_regions()
    return []  # Fail safe, caller may not check.


def aws_conn_getinfo(conn):
    """ Connection region name """
    region_name = None
    if conn:
        print(conn.region.name)
        region_name = conn.region.name
    return region_name


def aws_ec2_ami_create(conn, ip_address='', ami_name=''):
    """ Createan an AMI for the instance represented by the ipadress"""
    if conn and ip_address:
        instance_list = conn.get_all_instances(
            filters={
                "ip_address": ip_address
            }
        )
        instance = instance_list[0].instances[0] if instance_list else None
        if instance:
            ami_id = instance.create_image(ami_name)
            print("Image creation(%s) for AMI ID %s started...." % (ami_name, ami_id))
            return True
    return False


def aws_ec2_get_instanceid(conn, filters=None):
    """ Get Instance Id based on the filters. """
    if not filters:
        filters = {
            "ip_address": ""
        }
    if conn:
        instance_list = conn.get_all_instances(filters=filters)
        instance = instance_list[0].instances[0] if instance_list else None
        if instance:
            return instance.id
    return None


def aws_ec2_allocate_elastic_ip(conn, instance_id='', elastic_ip='', region=''):
    """ Allocate elastic IP in the region."""
    if not region:
        region = AWS().v['APNORTHEAST2']
    if conn and instance_id:
        if not elastic_ip:
            eip = conn.allocate_address()
            elastic_ip = eip.public_ip
        if elastic_ip:
            conn.associate_address(instance_id=instance_id, public_ip=elastic_ip)
            print('Elastic assigned Public IP: %s,Instance_ID: %s' % (elastic_ip, instance_id))
            return elastic_ip
    return None


def aws_ec2_allocate_eip(instance_id, conn=None, eip_allocation_id=None, eip_public_ip=None,
                         allow_reassociation=False):
    """
    Assign an Elastic IP to an instance. Works with either by specifying the
    connection: EC2Connection object
    instance_id: Desired EC2 instance's ID
    eip_allocation_id: ID of Elastic IP to assign (Required if no public_ip)
    eip_public_ip: Elastic IP to assign (Required if no allocation_id)
    allow_reassociation: Option to turn off reassociation (check caveats below)
    """
    if conn:
        if eip_public_ip:
            conn.associate_address(instance_id=instance_id, public_ip=eip_public_ip,
                                   allow_reassociation=allow_reassociation)
            return True
        elif eip_allocation_id:
            conn.associate_address(instance_id=instance_id, allocation_id=eip_allocation_id,
                                   allow_reassociation=allow_reassociation)
            return True
        else:
            raise ValueError("eip_public_ip and eip_allocation_id cannot be both None!")


def aws_ec2_spot_start(conn, region, key_name="ecsInstanceRole", inst_type="cx2.2", ami_id="",
                       pricemax=0.15, elastic_ip='', pars=None):
    """
    :param conn: Connector to Boto
    :param region: AWS region (us-east-1,..)
    :param key_name: AWS SSH Key Name(in EC2 webspage )
    :param security_group: AWS security group id
    :param inst_type:  AWS EC2 instance type(t1.micro, m1.small ...)
    :param ami_id: AWS AMI ID
    :param pars: Disk Size, Volume type (General Purpose SSD - gp2, Magnetic etc)
    :param pricemax: minmum spot instance bid price
    """
    if not conn:
        return None
    if not pars:
        pars = {
            "security_group": [""],
            "disk_size": 25,
            "disk_type": "ssd",
            "volume_type": "gp2"
        }
    pars = dict2(pars)   #Dict to Attribut Dict
    print("starting EC2 Spot Instance")

    try:
        block_map = BlockDeviceMapping()
        device = EBSBlockDeviceType()
        device.size = int(pars.disk_size)  # size in GB
        device.volume_type = pars.volume_type  # [ standard | gp2 | io1 ]
        device.delete_on_termination = False
        block_map["/dev/xvda"] = device
        print("created a block device")

        req = conn.request_spot_instances(price=pricemax, image_id=ami_id, instance_type=inst_type,
                                          key_name=key_name, security_groups=pars.security_group,
                                          block_device_map=block_map)
        print("Spot instance request created, status: %s" % (req[0].status))
        print("Waiting for spot instance provisioning")
        while True:
            current_req = conn.get_all_spot_instance_requests([req[0].id])[0]
            if current_req.state == "active":
                print("Spot instance provisioning successful.")
                instance = conn.get_all_instances([current_req.instance_id])[0].instances[0]
                aws_ec2_allocate_elastic_ip(conn, current_req.instance_id, elastic_ip, region)
                # Print Instance details : ID, IP adress
                print('IP Addess:%s, instance_id: %s' % (getattr(instance, "ip_address"),
                                                         getattr(instance, 'id')))
                return instance
            # print('.', end='')
            sleep(15)
    except:
        pass


def aws_ec2_spot_stop(conn, ipadress="", instance_id=""):
    """
    :param conn: connector
    :param ipadress:   of the instance  to Identify the instance.
    :param instance_id:  OR use instance ID....
    :return:
    """
    if not conn:
        return False
    if not instance_id:
        instance_id=  aws_ec2_get_instanceid(conn, ipadress)  # Get ID from IP Adresss
    try:
        print("Terminating Spot Instance : %s" % (str(instance_id)))
        conn.terminate_instances(instance_ids = [instance_id])
        print ("Terminating spot instance Successful!")
        return True
    except Exception as e:
        print ("Error : Failed to terminate. %s" % (str(e)))
    return False


def aws_ec2_res_start(conn, region, key_name, ami_id, inst_type="cx2.2",
                      min_count =1, max_count=1, pars=None):
    """
    normal instance start
    :param conn: Connector to Boto
    :param region: AWS region (us-east-1,..)
    :param key_name: AWS  SSH Key Name
    :param security_group: AWS security group id
    :param inst_type:  AWS EC2 instance type (t1.micro, m1.small ...)
    :param ami_id:  AWS AMI ID
    :param min_count: Minumum number of instances
    :param max_count : Maximum number of instances
    :param pars: Disk Size, Volume type (General Purpose SSD - gp2,Magnetic etc)
    :return
    """
    if not conn:
        return None
    if not pars:
        pars = {
            "security_group": [""],
            "disk_size": 25,
            "disk_type": "ssd",
            "volume_type": "gp2"
        }
    pars = dict2(pars)   #Dict to Attribut Dict
    try: 
        block_map = BlockDeviceMapping()
        device = EBSBlockDeviceType()
        device.size = int(pars.disk_size)  # size in GB
        device.volume_type = pars.volume_type  # [ standard | gp2 | io1 ]
        device.delete_on_termination = False
        block_map["/dev/xvda"] = device
        print("created a block device")
        req = conn.run_instances(image_id=ami_id, min_count=min_count,
                                 max_count=max_count,
                                 instance_type=inst_type, key_name=key_name,
                                 security_groups= pars.security_group,
                                 block_device_map=block_map)
        instance_id = req.instances[0].id
        print("EC2 instance has been created. Instance ID : %s" % instance_id)
        print("Waiting for EC2 instance provisioning")
        while True:
            print('.', end='')
            current_req = conn.get_all_instances(instance_id)
            if current_req[0].instances[0].state.lower() == "running":
                print("EC2 instance provisioning successful and the instance is running.")
                aws_ec2_printinfo(current_req[0].instances[0])
                return current_req[0].instances[0]
            print('.'),
            sleep(30)
    except Exception as e:
        print("Error : %s " % (str(e)))
    return None


def aws_ec2_res_stop(conn, ipadress="", instance_id=""):
    """
    :param conn: connector
    :param ipadress:     Of the instance  to Identify the instance.
    :param instance_id:  OR use instance ID....
    """
    if not conn:
        return False
    if not instance_id:
        instance_id = aws_ec2_get_instanceid(conn, ipadress)
    try:
        print("Stopping EC2 Instance : %s" % (str(instance_id)))
        req = conn.stop_instances(instance_ids=[instance_id])
        print("EC2 instance has been stopped successfully. %s" % req)
        return True
    except Exception as e:
        print("Error : Failed to stop EC2 instance. %s" % str(e))
    return False


def aws_ec2_printinfo(instance=None, ipadress="", instance_id=""):
    """
    Idenfiy instnance of
    :param instance: ipadress
    :param instance_id:
    :return: return info on the instance : ip, ip_adress,
    """
    if not ipadress:
        print("")
    if not instance_id:
        pass


def aws_ec2_get_folder(ipadress, fromfolder1, tofolder1):
    """Copy Folder from remote to  EC2 folder"""
    pass


def aws_s3_url_split(url):
    """Split into Bucket, url."""
    url_parts = url.split("/")
    return url_parts[0], "/".join(url_parts[1:])


def aws_s3_getbucketconn(s3dir):
    """ Get S3 bucket. """
    import boto.s3
    bucket_name, todir = aws_s3_url_split(s3dir)
    access, secret = AWS().aws_accesskey_get()
    conn = boto.connect_s3(access, secret)
    bucket = conn.get_bucket(bucket_name)  # location=boto.s3.connection.Location.DEFAULT)
    return bucket


def aws_s3_put(fromdir_file='dir/file.zip', todir='bucket/folder1/folder2'):
    """
    Copy File or Folder to S3
    "aws s3 cp s3://s3-bucket/scripts/HelloWorld.sh /home/ec2-user/HelloWorld.sh",
    "chmod 700 /home/ec2-user/HelloWorld.sh",
    "/home/ec2-user/HelloWorld.sh"
    """
    def percent_cb(complete, total):
        sys.stdout.write('.'); sys.stdout.flush()

    import boto.s3
    bucket = aws_s3_getbucketconn(todir)
    bucket_name, todir = aws_s3_url_split(todir)

    MAX_SIZE = 20 * 1000 * 1000
    PART_SIZE = 6 * 1000 * 1000

    if fromdir_file.find('.') > -1:   # Case of Single File
        filename = util.os_file_getname(fromdir_file)
        fromdir_file = util.os_file_getpath(fromdir_file) + '/'
        uploadFileNames = [filename]
    else:
        uploadFileNames = []
    for (fromdir_file, dirname, filename) in os.walk(fromdir_file):
        uploadFileNames.extend(filename)
        break

    for filename in uploadFileNames:
        sourcepath = os.path.join(fromdir_file + filename)
        destpath = os.path.join(todir, filename)
        print("Uploading %s to Amazon S3 bucket %s" % (sourcepath, bucket_name))
        filesize = os.path.getsize(sourcepath)
        if filesize > MAX_SIZE:
            print("multipart upload")
            mp = bucket.initiate_multipart_upload(destpath)
            fp = open(sourcepath, 'rb')
            fp_num = 0
            while fp.tell() < filesize:
                fp_num += 1
                print("uploading part %i" %fp_num)
                mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10,
                                         size=PART_SIZE)
            mp.complete_upload()
        else:
            print("singlepart upload: " + fromdir_file + ' TO ' + todir)
            k = boto.s3.key.Key(bucket)
            k.key = destpath
            k.set_contents_from_filename(sourcepath, cb=percent_cb, num_cb=10)


def aws_s3_get(froms3dir='task01/', todir='', bucket_name='zdisk'):
    """Get from S3 file/folder"""
    bucket_name, dirs3 = aws_s3_url_split(froms3dir)
    bucket = aws_s3_getbucketconn(froms3dir)
    bucket_list = bucket.list(prefix=dirs3)  # /DIRCWD/dir2/dir3

    for l in bucket_list:
        key1 = str(l.key)
        file1, path2 = util.os_file_getname(key1), util.os_file_getpath(key1)
        # Remove prefix path of S3 to machine
        path1 = os.path.relpath(path2, dirs3).replace('.', '')
        d = todir + '/' + path1
        if not os.path.exists(d):
            os.makedirs(d)
        try:
            l.get_contents_to_filename(d+ '/'+ file1)
        except OSError:
            pass


def aws_s3_folder_printtall(bucket_name='zdisk'):
    """ Print all S3 bucket folders"""
    access, secret = AWS().aws_accesskey_get()
    conn = boto.connect_s3(access, secret)
    bucket = conn.create_bucket(bucket_name,
                                location=boto.s3.connection.Location.DEFAULT)
    folders = bucket.list("", "/")
    for folder in folders:
        print(folder.name)
    return folders


def aws_s3_file_read(bucket1, filepath):
    """
    s3_client = boto3.client('s3')
    # Download private key file from secure S3 bucket
    s3_client.download_file('s3-key-bucket','keys/keyname.pem',
                            '/tmp/keyname.pem')
    """
    from boto.s3.connection import S3Connection
    conn = S3Connection(AWS().aws_accesskey_get())
    response = conn.get_object(Bucket=bucket1, Key=filepath)
    file1 = response["Body"]
    return file1


class aws_ec2_ssh(object):
    """
    ssh= aws_ec2_ssh(host)
    print ssh.command('ls ')
    ssh.put_all( DIRCWD +'linux/batch/task/elvis_prod_20161220', EC2CWD + '/linux/batch/task' )
    ssh.get_all(  EC2CWD + '/linux/batch/task',  DIRCWD +'/zdisks3/fromec2' )
    # Detects DSA or RSA from key_file, either as a string filename or a file object.
    # Password auth is possible, but I will judge you for
    # ssh=SSHSession('targetserver.com','root',key_file=open('mykey.pem','r'))
    # ssh=SSHSession('targetserver.com','root',key_file='/home/me/mykey.pem')
    # ssh=SSHSession('targetserver.com','root','mypassword')
    # ssh.put('filename','/remote/file/destination/path')
    # ssh.put_all('/path/to/local/source/dir','/path/to/remote/destination')
    # ssh.get_all('/path/to/remote/source/dir','/path/to/local/destination')
    # ssh.command('echo "Command to execute"')
    """

    def __init__(self, hostname, username='ubuntu', key_file=None, password=None):
        import socket
        # Accepts a file-like object (anything with a readlines() function)
        # in either dss_key or rsa_key with a private key.  Since I don't
        # ever intend to leave a server open to a password auth.
        self.host = hostname
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname,22))
        self.t = paramiko.Transport(self.sock)
        self.t.start_client()
        # keys = paramiko.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        # key = self.t.get_remote_server_key()
        # supposed to check for key in keys, but I don't much care right now to
        # find the right notation

        key_file = AWS().v['AWS_KEYPEM'] if not key_file else key_file
        pkey = paramiko.RSAKey.from_private_key_file(key_file)
        self.t.auth_publickey(username, pkey)
        self.sftp = paramiko.SFTPClient.from_transport(self.t)
        print(self.command('ls '))

    def command(self, cmd):
        # Breaks the command by lines, sends and receives
        # each line and its output separately
        # Returns the server response text as a string
        chan = self.t.open_session()
        chan.get_pty()
        chan.invoke_shell()
        chan.settimeout(20.0)
        ret = ''
        try:
            ret += chan.recv(1024)
        except:
            chan.send('\n')
            ret += chan.recv(1024)
        for line in cmd.split('\n'):
            chan.send(line.strip() + '\n')
            ret += chan.recv(1024)
        return ret

    def cmd(self, cmdss):
        ss = self.command(cmdss)
        print(ss)

    def put(self, localfile, remotefile):
        #  Copy localfile to remotefile, overwriting or creating as needed.
        self.sftp.put(localfile, remotefile)

    def put_all(self, localpath, remotepath):
        # recursively upload a full directory
        # localpath= localpath[:-1] if localpath[-1]=='/' else localpath
        # remotepath= remotepath[:-1] if remotepath[-1]=='/' else remotepath
        os.chdir(os.path.split(localpath)[0])
        parent = os.path.split(localpath)[1]
        print(parent)
        for walker in os.walk(parent):
            try:
                self.sftp.mkdir(os.path.join(remotepath, walker[0]).replace('\\', "/"))
            except:
                pass
            for file in walker[2]:
                print(os.path.join(walker[0], file).replace('\\', '/').replace('\\', '/'),
                      os.path.join(remotepath, walker[0], file).replace('\\', '/'))
                self.put(os.path.join(walker[0], file).replace('\\', '/'),
                         os.path.join(remotepath, walker[0], file).replace('\\', '/'))

    def get(self, remotefile, localfile):
        #  Copy remotefile to localfile, overwriting or creating as needed.
        self.sftp.get(remotefile, localfile)

    def sftp_walk(self, remotepath):
        from stat import S_ISDIR
        # Kindof a stripped down  version of os.walk, implemented for
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path = remotepath
        files = []
        folders = []
        for f in self.sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        print(path, folders, files)
        yield path, folders, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder).replace('\\', '/')
            for x in self.sftp_walk(new_path):
                yield x

    def get_all(self, remotepath, localpath):
        #  recursively download a full directory
        #  Harder than it sounded at first, since paramiko won't walk
        # For the record, something like this would gennerally be faster:
        # ssh user@host 'tar -cz /source/folder' | tar -xz
        localpath = localpath[:-1] if localpath[-1] == '/' else localpath
        remotepath = remotepath[:-1] if remotepath[-1] == '/' else remotepath

        self.sftp.chdir(os.path.split(remotepath)[0])
        parent = os.path.split(remotepath)[1]
        try:
            os.mkdir(localpath)
        except:
            pass
        for walker in self.sftp_walk(parent):
            try:
                os.mkdir(os.path.join(localpath, walker[0]).replace('\\', '/'))
            except:
                pass
            for file in walker[2]:
                print(os.path.join(walker[0], file).replace('\\', '/'),
                      os.path.join(localpath, walker[0], file).replace('\\', '/'))
                self.get(os.path.join(walker[0], file).replace('\\', '/'),
                         os.path.join(localpath, walker[0], file).replace('\\', '/'))

    def write_command(self, text, remotefile):
        # Writes text to remotefile, and makes remotefile executable.
        # This is perhaps a bit niche, but I was thinking I needed it.
        # For the record, I was incorrect.
        self.sftp.open(remotefile, 'w').write(text)
        self.sftp.chmod(remotefile, 755)

    def python_script(self, ipython_path='/home/ubuntu/anaconda3/bin/ipython ',
                      script_path="", args1=""):
        cmd1 = ipython_path + ' ' + script_path + ' ' + '"' + args1 +'"'
        self.cmd2(cmd1)
        # self.command(cmd1)

    def command_list(self, cmdlist):
        for command in cmdlist:
            print("Executing {}".format(command))
            ret = self.command(command)
            print(ret)
        print('End of SSH Command')

    def listdir(self, remotedir):
        return self.sftp.listdir(remotedir)

    def jupyter_kill(self):
        pid_jupyter = aws_ec2_ssh_cmd(cmdlist=['fuser 8888/tcp'], host=self.host,
                                      doreturn=1)[0][0].strip()
        print(self.command('kill -9 ' + pid_jupyter))

    def jupyter_start(self):
        pass

    def cmd2(self, cmd1):
        return aws_ec2_ssh_cmd(cmdlist=[cmd1], host=self.host, doreturn=1)

    def _help_ssh(self):
        s = """
           fuser 8888/tcp     Check if Jupyter is running
           ps -ef | grep python     :List of  PID Python process
           kill -9 PID_number     (i.e. the pid returned)
           top: CPU usage
          """
        print(s)

    def put_all_zip(self, suffixfolder="", fromfolder="", tofolder="",
                    use_relativepath=True, usezip=True, filefilter="*.*",
                    directorylevel=1, verbose=0):
        """
        fromfolder:  c:/folder0/folder1/folder2/*
        suffixfolder: /folders1/folder2/
        tofolder: /home/ubuntu/myproject1/
        """
        fromfolder = fromfolder if fromfolder[-1] != '/' else  fromfolder[:-1]
        tmpfolder = '/'.join(fromfolder.split('/')[:-1]) + "/"
        zipname = fromfolder.split('/')[-1] + ".zip"
        print(tmpfolder, zipname)
        zipath = tmpfolder + zipname

        print("zipping")
        filezip = util.os_zipfolder(dir_tozip=fromfolder, zipname=zipath,
                                    dir_prefix=True, iscompress=True)
        print("ssh on remote")
        remote_zipath = os.path.join(tofolder, zipname).replace('\\', '/')
        self.put(filezip, remote_zipath)

        if usezip:
            print("Unzip on remote " + self.host)
            cmd = '/usr/bin/unzip ' + remote_zipath + ' -d ' + tofolder
            ss = self.command(cmd)
            if verbose:
                print(ss)

def aws_ec2_ssh_create_con(contype='sftp/ssh', host='ip', port=22,
                           username='ubuntu', keyfilepath='', password='',
                           keyfiletype='RSA', isprint=1):
    """
    Transfert File  host = '52.79.79.1'
    keyfilepath = 'D:/_devs/aws/keypairs/ec2_instanc'
    # List files in the default directory on the remote computer.
    dirlist = sftp.listdir('.')
    sftp.get('remote_file.txt', 'downloaded_file.txt')
    sftp.put('testfile.txt', 'remote_testfile.txt')
    http://docs.paramiko.org/en/2.1/api/sftp.html
    """
    import paramiko
    sftp, ssh, transport = None, None, None
    try:
        if not keyfilepath:
            keyfilepath = AWS().v['AWS_KEYPEM']
        if keyfiletype == 'DSA':
            key = paramiko.DSSKey.from_private_key_file(keyfilepath)
        else:
            key = paramiko.RSAKey.from_private_key_file(keyfilepath)

        if contype == 'sftp':
            # Create Transport object using supplied method of authentication.
            transport = paramiko.Transport((host, port))
            transport.add_server_key(key)
            transport.connect(None, username, pkey=key)
            sftp = paramiko.SFTPClient.from_transport(transport)
            if isprint:
                print('Root Directory : %s\n' % sftp.listdir())
            return sftp

        if contype == 'ssh':
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, username=username, pkey=key)

            # Test
            if isprint:
                stdin, stdout, stderr = ssh.exec_command("uptime;ls -l")
                stdin.flush()   #Execute
                data = stdout.read().splitlines()   #Get data
                print('Test Print Directory ls :')
                for line in data: print(line)
                return ssh

    except Exception as e:
        print('An error occurred creating client: %s: %s' % (e.__class__, e))
        if sftp: sftp.close()
        if transport: transport.close()
        if ssh: ssh.close()
    return None


def aws_ec2_ssh_cmd(cmdlist=["ls "], host='ip', doreturn=0, ssh=None,
                    username='ubuntu', keyfile=''):
    """
    SSH Linux terminal Command
    ### PEM File is needed
    aws_ec2_ssh_cmd(cmdlist=["ls " ], host='52.26.181.200', doreturn=1,
                    username='ubuntu', keyfilepath='')
    fuser 8888/tcp - Check if Jupyter is running
    ps -ef | grep python : List of  PID Python process
    kill -9 PID_number (i.e. the pid returned)
    top: CPU usage

    Run nohup python bgservice.py & to get the script to ignore the hangup signal and keep running.
    Output will be put in nohup.out.
    https://aws.amazon.com/blogs/compute/scheduling-ssh-jobs-using-aws-lambda/
    """


    def command(self, cmd):
        # Breaks the command by lines, sends and receives
        # each line and its output separately
        # Returns the server response text as a string
        chan = self.t.open_session()
        chan.get_pty()
        chan.invoke_shell()
        chan.settimeout(20.0)
        ret = ''
        try:
            ret += chan.recv(1024)
        except:
            chan.send('\n')
            ret += chan.recv(1024)
        for line in cmd.split('\n'):
            chan.send(line.strip() + '\n')
            ret += chan.recv(1024)
        return ret

    def cmd(self, cmdss):
        ss = self.command(cmdss)
        print(ss)

    def put(self, localfile, remotefile):
        #  Copy localfile to remotefile, overwriting or creating as needed.
        self.sftp.put(localfile, remotefile)

    def put_all(self, localpath, remotepath):
        # recursively upload a full directory
        # localpath= localpath[:-1] if localpath[-1]=='/' else localpath
        # remotepath= remotepath[:-1] if remotepath[-1]=='/' else remotepath
        os.chdir(os.path.split(localpath)[0])
        parent = os.path.split(localpath)[1]
        print(parent)
        for walker in os.walk(parent):
            try:
                self.sftp.mkdir(os.path.join(remotepath, walker[0]).replace('\\', "/"))
            except:
                pass
            for file in walker[2]:
                print(os.path.join(walker[0], file).replace('\\', '/').replace('\\', '/'),
                      os.path.join(remotepath, walker[0], file).replace('\\', '/'))
                self.put(os.path.join(walker[0], file).replace('\\', '/'),
                         os.path.join(remotepath, walker[0], file).replace('\\', '/'))

    def get(self, remotefile, localfile):
        #  Copy remotefile to localfile, overwriting or creating as needed.
        self.sftp.get(remotefile, localfile)

    def sftp_walk(self, remotepath):
        from stat import S_ISDIR
        # Kindof a stripped down  version of os.walk, implemented for
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path = remotepath
        files = []
        folders = []
        for f in self.sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        print(path, folders, files)
        yield path, folders, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder).replace('\\', '/')
            for x in self.sftp_walk(new_path):
                yield x

    def get_all(self, remotepath, localpath):
        #  recursively download a full directory
        #  Harder than it sounded at first, since paramiko won't walk
        # For the record, something like this would gennerally be faster:
        # ssh user@host 'tar -cz /source/folder' | tar -xz
        localpath = localpath[:-1] if localpath[-1] == '/' else localpath
        remotepath = remotepath[:-1] if remotepath[-1] == '/' else remotepath

        self.sftp.chdir(os.path.split(remotepath)[0])
        parent = os.path.split(remotepath)[1]
        try:
            os.mkdir(localpath)
        except:
            pass
        for walker in self.sftp_walk(parent):
            try:
                os.mkdir(os.path.join(localpath, walker[0]).replace('\\', '/'))
            except:
                pass
            for file in walker[2]:
                print(os.path.join(walker[0], file).replace('\\', '/'),
                      os.path.join(localpath, walker[0], file).replace('\\', '/'))
                self.get(os.path.join(walker[0], file).replace('\\', '/'),
                         os.path.join(localpath, walker[0], file).replace('\\', '/'))

    def write_command(self, text, remotefile):
        # Writes text to remotefile, and makes remotefile executable.
        # This is perhaps a bit niche, but I was thinking I needed it.
        # For the record, I was incorrect.
        self.sftp.open(remotefile, 'w').write(text)
        self.sftp.chmod(remotefile, 755)

    def python_script(self, ipython_path='/home/ubuntu/anaconda3/bin/ipython ',
                      script_path="", args1=""):
        cmd1 = ipython_path + ' ' + script_path + ' ' + '"' + args1 +'"'
        self.cmd2(cmd1)
        # self.command(cmd1)

    def command_list(self, cmdlist):
        for command in cmdlist:
            print("Executing {}".format(command))
            ret = self.command(command)
            print(ret)
        print('End of SSH Command')

    def listdir(self, remotedir):
        return self.sftp.listdir(remotedir)

    def jupyter_kill(self):
        pid_jupyter = aws_ec2_ssh_cmd(cmdlist=['fuser 8888/tcp'], host=self.host,
                                      doreturn=1)[0][0].strip()
        print(self.command('kill -9 ' + pid_jupyter))

    def jupyter_start(self):
        pass

    def cmd2(self, cmd1):
        return aws_ec2_ssh_cmd(cmdlist=[cmd1], host=self.host, doreturn=1)

    def _help_ssh(self):
        s = """
           fuser 8888/tcp     Check if Jupyter is running
           ps -ef | grep python     :List of  PID Python process
           kill -9 PID_number     (i.e. the pid returned)
           top: CPU usage
          """
        print(s)

    def put_all_zip(self, suffixfolder="", fromfolder="", tofolder="",
                    use_relativepath=True, usezip=True, filefilter="*.*",
                    directorylevel=1, verbose=0):
        """
        fromfolder:  c:/folder0/folder1/folder2/*
        suffixfolder: /folders1/folder2/
        tofolder: /home/ubuntu/myproject1/
        """
        fromfolder = fromfolder if fromfolder[-1] != '/' else  fromfolder[:-1]
        tmpfolder = '/'.join(fromfolder.split('/')[:-1]) + "/"
        zipname = fromfolder.split('/')[-1] + ".zip"
        print(tmpfolder, zipname)
        zipath = tmpfolder + zipname

        print("zipping")
        filezip = util.os_zipfolder(dir_tozip=fromfolder, zipname=zipath,
                                    dir_prefix=True, iscompress=True)
        print("ssh on remote")
        remote_zipath = os.path.join(tofolder, zipname).replace('\\', '/')
        self.put(filezip, remote_zipath)

        if usezip:
            print("Unzip on remote " + self.host)
            cmd = '/usr/bin/unzip ' + remote_zipath + ' -d ' + tofolder
            ss = self.command(cmd)
            if verbose:
                print(ss)
    if ssh is None and len(host) > 5:
        ssh = aws_ec2_ssh_create_con(contype='ssh', host=host, port=22, username=username,
                                     keyfilepath='')
        print('EC2 connected')

    c = cmdlist
    if isinstance(c, str):  # Only Command to be launched
        if c == 'python': cmfile = AWS().v['AWS_KEYPEM'] if not keyfile else keyfile
        pkey = paramiko.RSAKey.from_private_key_file(keyfile)
        ssh.t.auth_publickey(username, pkey)
        ssh.sftp = paramiko.SFTPClient.from_transport(ssh.t)
        print(command('ls '))


def aws_ec2_ssh_python_script(python_path='/home/ubuntu/anaconda2/bin/ipython',
                              script_path="", args1="", host=""):
   # No space after ipython
    cmd1 = python_path + ' ' + script_path + ' ' + '"' + args1 + '"'
    aws_ec2_ssh_cmd(cmdlist=[cmd1], ssh=None, host=host, username='ubuntu')


def aws_ec2_get_instances(con=None, attributes=None, filters=None, csv_filename=".csv"):
    """
    Fetch all EC2 instances and write selected attributes into a csv file.
    Parameters:
      connection: EC2Connection object
      attributes: Tuple of attributes to retrieve. Default: id, ip_address
      filters={"ip_address":""}
    """
    if not attributes: attributes = AWS().v['EC2_ATTRIBUTES']
    instances = con.get_only_instances(filters=filters)
    instance_list = []
    for instance in instances:
        # attribute is a python keyword.
        x = {attr: getattr(instance, attr) for attr in attributes}
        instance_list.append(x)
    try:
        if len(csv_filename) > 4:
            with open(csv_filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",")
                csvwriter.writerow(list(attributes))
                for xi in instance_list:
                    csvwriter.writerow([xi[t] for t in attributes])
            print("EC2 instance", csv_filename)
    except Exception as e:
        print(e)
    return instance_list


def aws_ec2_getfrom_ec2(fromfolder, tofolder, host):
    sftp = aws_ec2_ssh_create_con(contype='sftp', host=host)
    if fromfolder.find('.') > -1:  # file
        folder1, file1 = util.z_key_splitinto_dir_name(fromfolder[:-1] if fromfolder[-1] == '/' else fromfolder)
        tofolder2 = tofolder if tofolder.find(".") > -1 else  tofolder + '/' + file1
        sftp.get(fromfolder, tofolder2)
    else:  # Pass the Folder in Loop
        pass


def aws_ec2_putfolder(fromfolder='D:/_d20161220/', tofolder='/linux/batch', host=''):
    # fromfolder= DIRCWD +'/linux/batch/task/elvis_prod_20161220/'
    # tofolder=   '/linux/batch/'
    # If you don't care whether the file already exists and you always want to overwrite the
    # files as they are extracted without prompting use the -o switch as follows:
    # https://www.lifewire.com/examples-linux-unzip-command-2201157
    # unzip -o filename.zip
    aws = AWS().v
    folder1, file1 = util.z_key_splitinto_dir_name(fromfolder[:-1] if fromfolder[-1]=='/' else fromfolder)
    tofolderfull = aws['EC2CWD'] + '/' + tofolder if tofolder.find(aws['EC2CWD']) == -1 else tofolder

    # Zip folder before sending it
    file2 = folder1 + '/' + file1 + '.zip'
    util.os_zipfolder(fromfolder, file2)
    res = aws_ec2_putfile(file2, tofolder=tofolderfull, host=host)
    print(res)

    # Need install sudo apt-get install zip unzip
    cmd1 = '/usr/bin/unzip '+ tofolderfull + '/' + file1 + '.zip ' + ' -d ' +  tofolderfull + '/'
    aws_ec2_ssh_cmd(cmdlist=[cmd1], host=host)


def aws_ec2_put(fromfolder='d:/file1.zip', tofolder='/home/notebook/aapackage/', host='',
                typecopy='code'):
    """
    Copy python code, copy specific file, copy all folder content
    :param fromfolder: 1 file or 1 folder
    :param tofolder:
    :param host:
    """
    DIRCWD  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fromfolder.find('.') > -1:  # Copy 1 file
        aws_ec2_putfile(fromfolder=fromfolder, tofolder=tofolder, host=host)
    else:  # Copy Folder to
        sftp = aws_ec2_ssh_create_con('sftp', host, isprint=1)
        # Local folder and code folder
        if typecopy == 'code' and fromfolder.find('.') == -1:
            # foldername = fromfolder
            # fromfolder = DIRCWD+ '/' + foldername
            tempfolder = DIRCWD + '/ztemp/' + fromfolder
            # TODO os_folder_delete to be added in util.py
            util.os_folder_delete(tempfolder)
            util.os_folder_copy(fromfolder, tempfolder, pattern1="*.py")
            sftp.put(tempfolder, tofolder)
            return 1
        if typecopy == 'all':
            if fromfolder.find(':') == -1:
                print('Please put absolute path')
                return 0
            if fromfolder.find('.') > -1:  #1 file
                fromfolder, file1 = util.os_split_dir_file(fromfolder)
                tofull = tofolder + '/' + file1 if tofolder.find('.') == -1 else tofolder
                tofolder, file2 = util.os_split_dir_file(tofull)
            sftp.put(fromfolder + '/' + file1, tofull)
            try:
                sftp.stat(tofull)
                isexist = True
            except:
                isexist = False
            print(isexist, tofull)


def aws_ec2_putfile(fromfolder='d:/file1.zip', tofolder='/home/notebook/aapackage/',
                    host=''):
    """
    Copy python code, copy specific file, copy all folder content
    :param fromfolder: 1 file or 1 folder
    :param tofolder:
    :param host:
    """
    sftp = aws_ec2_ssh_create_con('sftp', host, isprint=1)
    if fromfolder.find('.') > -1:  # Copy 1 file
        if fromfolder.find(':') == -1:
            print('Please put absolute path')
            return 0
    fromfolder2, file1 = util.z_key_splitinto_dir_name(fromfolder)
    tofull = tofolder + '/' + file1 if tofolder.find('.') == -1 else tofolder
    print("from: %s, to: %s" % (fromfolder, tofull))
    isexist = False
    try:
        sftp.put(fromfolder, tofull)
        ss = sftp.stat(tofull)
        isexist = True
    except Exception as e:
        print(e)
        ss = None
    sftp.close()
    return isexist, fromfolder, tofull, ss


def sleep2(wsec):
    from time import sleep
    from tqdm import tqdm
    for i in tqdm(range(wsec)):
        sleep(1)


def sftp_isdir(path, sftp):
    from stat import S_ISDIR
    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        #Path does not exist, so by definition not a directory
        return False


def aws_ec2_getfolder(remotepath, sftp):
    import paramiko, os
    paramiko.util.log_to_file('/tmp/paramiko.log')
    from stat import S_ISDIR
    def sftp_walk(remotepath):
        path = remotepath
        files = []
        folders = []
        for f in sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        if files:
            yield path, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder)
            for x in sftp_walk(new_path):
                yield x
    for path, files in sftp_walk("." or '/remotepath/'):
        for file1 in files:
            #sftp.get(remote, local) line for dowloading.
            sftp.get(os.path.join(os.path.join(path, file1)), '/local/path/')



####################################################################################################
############################ UTILS #################################################################
class dict2(object):
    # {} INTO   mydict.key1   ,  mydict.key2 ,   mydict.key4
    def __init__(self, adict):
       self.__dict__.update(adict)

def os_system(cmds, stdout_only=1) :
  #   Get print output from command line
  import subprocess
  cmds = cmds.split(" ")
  p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = p.stdout.read(),  p.stderr.read()

  if stdout_only :
        return out
  return out, err


def tofloat(value, default=0.0):
  #  Parse the float value.
  try:
    return float(value)
  except:
    return default



### Remove util dependencies
def os_file_getname(fromdir_file) :
    pass


def os_file_getpath(fromdir_file) :
    pass


def os_zipfolder(dir_tozip="", zipname="",       dir_prefix=True, iscompress=True) :
    pass


def  z_key_splitinto_dir_name(key) :
    pass


def os_folder_delete(tempfolder) :
    pass

def os_folder_copy(fromfolder, tempfolder, pattern1="*.py") :
    pass


####################################################################################################
############################ CLI ###################################################################
def cli_windows_start_spot():
        # :\_devs\Python01\aws\aapackage\
        # D:\_devs\Python01\ana27\python D:\_devs\Python01\aws\aapackage\util_aws.py --do start_spot
        # aws_ec2_spot_start(EC2_CONN, "west-2", key_name="ecsInstanceRole", inst_type="cx2.2",
        #                    ami_id="", pricemax=0.15,  elastic_ip='',
        #                    pars={"security_group": [""], "disk_size": 25, "disk_type": "ssd",
        #                         "volume_type": "gp2"})
        ss  = 'aws ec2 request-spot-instances --region us-west-2 --spot-price "0.55" --instance-count 1 '
        ss += ' --type "one-time" --launch-specification "file://D:\_devs\Python01\\awsdoc\\ec_config2.json" '
        print(ss)
        os.system(ss)
        sleep2(65)
        aws_win_conn = AWS().aws_conn_create_windows()
        ec2_list = aws_ec2_get_instances(aws_win_conn, csv_filename="zz_ec2_instance.csv")
        print(ec2_list)
        for x in ec2_list :
            if x["state"] == "running":
                sleep(5) 
                ss = ' start "Chrome" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" '
                ss += ' "http://' +  x["ip_address"] + ':8888/tree#notebooks"  '
                os.system(ss)    


def test_all():
    pass



if __name__ == '__main__':
    print("Start")
    import argparse
    p = argparse.ArgumentParser()  # Command Line input
    p.add_argument('--do', type=str, default='action', help='start_spot')
    p.add_argument('--price', type=float, default=0.5, help='spot price')
    p.add_argument('--spec_file', type=str, default="", help='spec file')
    p.add_argument('--spec_file2', type=str, default="", help='spec file')
    
    args = p.parse_args()
    print(args.do)
    
    if args.do == "test_all":
        test_all()
    
    if args.do == "start_spot_windows":
        cli_windows_start_spot()

    if args.do == "put_file":
        aws_ec2_putfile(fromfolder=args.fromfolder,
                        tofolder=args.tofolder, host=args.host)
