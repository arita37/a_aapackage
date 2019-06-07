# -*- coding: utf-8 -*-
%load_ext autoreload
%autoreload
import os
import sys
from subprocess import call

import numpy as np

import util

DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage');  sys.path.append(DIRCWD + '/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
###########################################################################################
# import subprocess, ConfigParser,  socket,  boto, pandas as pd
EC2CWD=      '/home/ubuntu/notebook/'
EC2_ipython= '/home/ubuntu/anaconda2/bin/'
params= {}



########################  Task on Spot Instance   ###################################################


#### 1) Create Task Folder,  Change  01_batch_param_generator.py  ####################################
params['task1_name']= 'elvis_prod_20160102'                     # Folder of  linux/batch/task/

DIRBATCH_local= DIRCWD + '/linux/batch/task/' + task1_name   #Local folder
task1_folder=   '/linux/batch/task/' + task1_name          #Remote Folder


'''
AWS_BATCH.py  --->
        subprocess_launcher_01  (sub-process)--->  pygmo_batch_generic.py ---> elvis_pygmo_optim.py
        using  ec2_config.py    all_input_params.pkl

01_batch_param_generator : Generate parameters for the compute into a file.

05_batch_getresults :   Transfer Folder EC2 ---> Local   + Close Instance when finished.


'''




#### 2) Create  Spot Instance  #############################################################
#    host=    '52.79.79.1'   #Static version

### 2a)   BE careful of AMI  when creating AMI !!!!!!!


### 2b)  Put the values below :
params['NCPU']=    2            #  16+5                        # 16 CPU: 80% Load
params['ec2_id']=  'i-05ecf49847f6e0905'       #Instance ID



params['host']=  util.aws_ec2_allocate_elastic_ip(instance_id=params['ec2_id'])
globals().update(params)
util.os_config_setfile(params, DIRBATCH_local + '/ec2_config.py', mode1='w+')
execfile(DIRBATCH_local + '/ec2_config.py')
print  '\n', 'task1_name:', task1_name, '\nNb CPU: ', NCPU, '\nHost:', host, '\nInstance_ID:', ec2_id

### Open the File Manually


#### 3) Install missing package ##########################################################
ssh= util.aws_ec2_ssh(host)


ssh.cmd2('/home/ubuntu/anaconda2/bin/conda install -y -c conda-forge pygmo=1.1.7')

ssh.cmd2('/home/ubuntu/anaconda2/bin/pip install kmodes')

ssh.cmd2('/home/ubuntu/anaconda2/bin/pip install ggplot')

ssh.cmd2('mkdir -p -v  notebook/linux/batch/task')

ssh.cmd2('mkdir -p -v  linux/batch/output')
print('Finished')





#### 4) Copy Local Folder to EC2      ####################################################
## Manual Copy of local aapackage to
# util.os_folder_copy(DIRCWD+'/aapacakge', DIRCWD+'/awsprod/aapackage', pattern1='*portfolio*.py')

ssh.put_all(DIRCWD +'/awsprod/aapackage', EC2CWD )

ssh.put_all(DIRBATCH_local, EC2CWD + '/linux/batch/task')     #  3) Copy Local  project27/aapackage  to awsprod/aapackage




#### 5) Launch Full Batch                ##############################################
''' ISSUES Some Process are killed when doing ssh if the number of core  is not enough

##  ipython /home/ubuntu/notebook/linux/batch/task/elvis_prod_20161228/batch_launcher_02.py

'''

####Check Params    -------------------------------------------------------------------
execfile(DIRBATCH_local + '/ec2_config.py')
print  '\n', 'task1_name:', task1_name, '\nNb CPU: ', NCPU, '\nHost:', host, '\nInstance_ID:', ec2_id


print(EC2CWD + task1_folder + '/subprocess_launcher_01.py')
ssh= util.aws_ec2_ssh(host)
ssh.python_script(EC2CWD + task1_folder + '/subprocess_launcher_01.py', 'args1')















##################################################################################################
##################################################################################################
##################################################################################################

'''
4.4x Large : 15 task: 85% Busy

'''



#### Retrieve Results from EC2 from output folder   ##################################
ssh.get_all( EC2CWD + '/linux/batch/output/',  DIRCWD +'/zdisks3/results/'+task1_name)




#### Loop to retrieve results   ########################################
while true :
   ssh.get_all( EC2CWD + '/linux/batch/output/',  DIRCWD +'/zdisks3/results/'+task1_name)
   time.sleep( 5* 60)  # Retrieve every 5mins













################  Amazon testing  in SSH #########################################################
host=    '52.79.79.1'

host= '52.79.134.19'
ssh= util.aws_ec2_ssh(host)


task1_name=   'elvis_prod_20161228'
task1_folder= '/linux/batch/task/' + task1_name


########################  Task 1   #######################################################
#### Copy Local Folder to EC2

ssh.put_all(DIRCWD +'/awsprod/aapackage', EC2CWD )

ssh.put_all(DIRCWD + task1_folder, EC2CWD + '/linux/batch/task' )


###  Modify the Batch_launcher FILE for number of CPU


#### Full Batch
##  ipython /home/ubuntu/notebook/linux/batch/task/elvis_prod_20161228/batch_launcher_02.py
ssh.python_script(EC2CWD + task1_folder+'/batch_launcher_02.py', 'args1')


'''
ISSUES Some Process are killed when doing manual ssh over windows if the number of core
is not enough
'''


#### Retrieve Results from EC2 from output folder
ssh.get_all( EC2CWD + '/linux/batch/output/',  DIRCWD +'/zdisks3/results/'+task1_name)


ssh.get_all( EC2CWD + '/linux/batch/output/20161231',  DIRCWD +'/zdisks3/results/'+task1_name)


util.os_zipfolder(DIRCWD +'/zdisks3/results/'+task1_name+'/', DIRCWD +'/zdisks3/results')



#Execute Script  Single Batch  GOOD !
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'
ssh.python_script(batch_folder+'pygmo_batch_02.py', 'args1')












txt_config= '''

host=         '52.79.134.19'
NCPU=          16
task1_name=   'elvis_prod_20161228'
task1_folder= '/linux/batch/task/' + task1_name

'''
util.os_print_tofile(txt_config, DIRBATCH_local + '/ec2_config.py', mode1='w+')
execfile(DIRBATCH_local + '/ec2_config.py') ;     print 'Nb CPU:', NCPU, 'host:', host, task1_name





#######################################################################################
#######################################################################################
###### Local Linux Launcher

ipython   /media/sf_project27/linux/batch/task/elvis_prod_20161228/batch_launcher_02.py


1) Generate the Params

2) Check launcher 02  :
     nb of CPU, hardware Split

3) Check batch generic config  :
    output results

4)  pygmo   :
   compute of one item



#### Remote
ipython /home/ubuntu/notebook//linux/batch/task/elvis_prod_20161228/batch_launcher_02.py









































##################################################################################################
################  Amazon testing  in SSH #########################################################


##################################################################################################

ssh= util.aws_ec2_ssh(host)
print ssh.command('ls ')

#### Copy Folder to EC2
ssh.put_all( DIRCWD +'linux/batch/task/elvis_prod_20161220', EC2CWD + '/linux/batch/task' )


#### Retrieve Folder from EC2
ssh.get_all(  EC2CWD + '/linux/batch/task',  DIRCWD +'/zdisks3/fromec2' )



#Execute Script  Single Batch  GOOD !
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'
ssh.python_script(batch_folder+'pygmo_batch_02.py', 'args1')



#### Retrieve Results from EC2
ssh.listdir(EC2CWD + '/linux/batch/output/20161228')

ssh.get_all(  EC2CWD + '/linux/batch/output/20161228',  DIRCWD +'/zdisks3/fromec2/new' )




########################  Full Batch    #######################################################
##  ipython /home/ubuntu/notebook//linux/batch/task/elvis_prod_20161220/batch_launcher_01.py
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'
ssh.python_script(batch_folder+'batch_launcher_01.py', 'args1')


'''
c4.4xlarge 	16 CPU  $0.1293  /  2h
Split into 16 batch !!!

Generate list of parameters :
Split into 15 parts

input_params_all()
 for
   for
      pi=
      input_params_all.append(pi)

#save in folder

0.10


for ii in xrange(i0, i1) :
  pi= input_params_all[ii,:]
  var1= pi[ii,:

  execfile()

4**6: 4096
256

48000 : combinaison of params
6 loops with 5 elelmets



'''


'''



######## Test Command  ##################################################################
print ssh.command('ps -ef | grep python ')


a= ssh.command('ps --sort=-pcpu | head -n 6')
print a



print ssh.command('top -b | head -n 8')



util.aws_ec2_cmd_ssh(cmdlist=  ['ps -ef | grep python '], host=host ,doreturn=1)




####################### Execute Script
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'

#Execute Script  Single Batch  GOOD !
util.aws_ec2_python_script(batch_folder+'pygmo_batch_02.py', 'args1', host)










#Start Jupyter in nohup mode
util.aws_ec2_jupyter(host)



#Copy Folder to Remote
util.aws_ec2_putfolder(fromfolder='D:/_devs/Python01/project27//linux/batch/task/elvis_prod_20161220/',
                  tofolder='/linux/batch/task/', host=host)


#Copy Files
util.aws_ec2_put(DIRCWD + '/linux/batch/task/elvis_prod_20161220.zip',
                 tofolder=EC2CWD + '/linux/batch/task/',
                 host=host, typecopy='all')

#UnZip
#   Need install sudo apt-get install zip unzip
foldertarget= '/linux/batch/task/elvis_prod_20161220/*'
file1=EC2CWD + 'linux/batch/task/elvis_prod_20161220.zip'
cmd= '/usr/bin/unzip '+ file1 + ' -d ' + EC2CWD + 'linux/batch/task/'
util.aws_ec2_cmd_ssh(cmdlist=  [cmd], host=host)



##################################################################################################
####################### Execute Script
batch_folder= EC2CWD + '/linux/batch/task/elvis_prod_20161220/'

#Execute Script  Single Batch  GOOD !
util.aws_ec2_python_script(batch_folder+'pygmo_batch_02.py', 'args1', host)


#Full Scripts
util.aws_ec2_python_script(batch_folder+'batch_launcher_01.py', 'args1', host)






lfile= [ (EC2CWD+'/linux/batch/output/20161228/batch_20161228_090648553369/', 'output_result.txt'),
 (EC2CWD+'/linux/batch/output/20161228/batch_20161228_090648553369/', 'aafolio_storage_20161228.pkl'),
]

sftp= util.aws_ec2_create_con('sftp', host)
for x in lfile :
  sftp.get( x[0] + '/' +x[1], DIRCWD+'/zdisks3/fromec2/'+ x[1])




##################################################################################################
#### SFTp Command ################################################
sftp= util.aws_ec2_create_con('sftp', host)

sftp.listdir(EC2CWD + '/linux/batch/task/')
sftp.mkdir(EC2CWD + '/newdir')
sftp.put(DIRCWD +'linux/batch/task/elvis_prod_20161220.zip', EC2CWD + 'linux/batch/task/elvis_prod_20161220.zip')

sftp.listdir()





###### SSH Command   ############################################
ssh= util.aws_ec2_create_con('ssh', host)

#Delete folder
rm *

#Check CPU Time
util.aws_ec2_cmd_ssh(cmdlist=  ['top '], host=host)


#Monitor the process
# top   : Monitor usage
# apt-get install sysstat
# https://www.cyberciti.biz/tips/how-do-i-find-out-linux-cpu-utilization.html




####################################################################################################

accepted
please check the below code from the link https://gist.github.com/johnfink8/2190472. I have used Put_all method in the below snippet




import
util.aws_ec2_jupyter(host)

print(a)

#########################################################################################################







CHUNKSIZE = 10240  # how much to read at a time
txt= proc.stdout.read(CHUNKSIZE)
print txt


################ Watcher on the processs
from subprocess import Popen, PIPE
from threading import Thread
from Queue import Queue, Empty

io_q = Queue()

def stream_watcher(identifier, stream):
    for line in stream:
        io_q.put((identifier, line))
    if not stream.closed:        stream.close()

proc = Popen('svn co svn+ssh://myrepo', stdout=PIPE, stderr=PIPE)


Thread(target=stream_watcher, name='stdout-watcher',   args=('STDOUT', proc.stdout)).start()
Thread(target=stream_watcher, name='stderr-watcher',   args=('STDERR', proc.stderr)).start()


def printer():
    while True:
        try:
            # Block for 1 second.
            item = io_q.get(True, 1)
        except Empty:
            # No output in either streams for a second. Are we done?
            if proc.poll() is not None:                break
        else:
            identifier, line = item
            print identifier + ':', line

Thread(target=printer, name='printer').start()




#############################################################################
#block the main
subprocess.call(["ipython", filescript ])

#No block the main
subprocess.Popen(["ipython", filescript] )


'''





##################################################################################################
######################  Initialization    #######################################################
INSTANCE_TYPE= ['t1.micro', 'm1.small', 'm1.medium', 'm1.large', 'm1.xlarge', 'm3.medium', 'm3.large', 'm3.xlarge', 'm3.2xlarge', 'c1.medium', 'c1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'cr1.8xlarge', 'hi1.4xlarge', 'hs1.8xlarge', 'cc1.4xlarge', 'cg1.4xlarge', 'cc2.8xlarge', 'g2.2xlarge', 'c3.large', 'c3.xlarge', 'c3.2xlarge', 'c3.4xlarge', 'c3.8xlarge', 'c4.large', 'c4.xlarge', 'c4.2xlarge', 'c4.4xlarge', 'c4.8xlarge', 'i2.xlarge', 'i2.2xlarge', 'i2.4xlarge', 'i2.8xlarge', 't2.micro', 't2.small', 't2.medium']
REGIONS = [("ap-northeast-2", "Asia Pacific (Seoul)"), ("ap-northeast-1", "Asia Pacific (Tokyo)"), ("ap-southeast-1", "Asia Pacific (Singapore)"), ("ap-southeast-2", "Asia Pacific (Sydney)"), ("eu-central-1", "EU (Frankfurt)"), ("eu-west-1", "EU (Ireland)"), ("sa-east-1", "South America (Sao Paulo)"), ("us-east-1", "US East (N. Virginia)"), ("us-west-1", "US West (N. California)"), ("us-west-2", "US West (Oregon)")]
DBNAME= DIR_package+'task_scheduler.db'

config= ConfigParser.ConfigParser()
config.read(DIR_package+'config.cfg')






with open('directory_of_logfile/logfile.txt', 'w') as f:
   call(['python', 'directory_of_called_python_file/called_python_file.py'], stdout=f)


'''

def aws_ec2_pythonscript():
   '''
   ps -ef | grep python     :List of  PID Python process
   kill -9 PID_number     (i.e. the pid returned)

   Run nohup python bgservice.py & to get the script to ignore the hangup signal and keep running. Output will be put in nohup.out.
   Ideally, you'd run your script with something like supervise so that it can be restarted if (when) it dies.


   %connect_info      : To get Connection Details of Ipytho notebook
   '''
   pass



subprocess.call(["robocopy", from_folder, to_folder, "/LOG:%s" % my_log])


    else:
        print("Paths not entered correctly")

'''
There are two ways to do the redirect. Both apply to either subprocess.Popen or subprocess.call.

Set the keyword argument shell = True or executable = /path/to/the/shell and specify the command just as you have it there.
Since you're just redirecting the output to a file, set the keyword argument

stdout = an_open_writeable_file_object
where the object points to the output file.
subprocess.Popen is more general than subprocess.call.

Popen doesn't block, allowing you to interact with the process while it's running, or continue with other things in your Python program. The call to Popen returns a Popen object.

call does block. While it supports all the same arguments as the Popen constructor, so you can still set the process' output, environmental variables, etc., your script waits for the program to complete, and call returns a code representing the process' exit status.

returncode = call(*args, **kwargs)
'''

https://pymotw.com/2/subprocess/

############################################################################################################
############################################################################################################

'''
def listener_ec2_spotprice():
   # Listener of SPot Price and Launche of EC2
   action=sys.argv[1]
   conn=aws_conn_create()
   if conn is None:
      print 'Unable to create EC2 ec2'; sys.exit(0)

   # inst= instance_get_existing(ec2)
   user_data=read_user_data_from_local_config()

   while true:
      spot_price= instance_spot_get_pricelast(conn)
      print 'Spot price is ' + str(spot_price) + ' ...'

      if spot_price > MAX_BID :
         print 'too high, wait 5mins'
         sleep(60 * 5)
      else:
         print 'below maximum bid, Creation of New Instance'
         instance_spot_request_start(conn, user_data)
         inst= instance_get_existing(conn)
         inst_ip= instance_wait_forup(conn, inst)  # Start the instance
         # db_instance_add(inst)           # No Need to wait for Instance            #  Add instance in the Database  or
'''

'''
# Get Keypair Name   -----------------------------------------------------------------------------------
rs= ec2.get_all_key_pairs()
for x in rs:  print x.name

#Create New Key-Pair
keypair = ec2.create_key_pair('ec2_newkeypair_2')
keypair.save('.')

# ec2.get_key_pair('ec2_instance_test01')
'''



############################################################################################################
ec2= aws_create_conn(region="ap-northeast-2")
print ec2.region.name

regions=ec2.get_all_regions()
for x in regions: print x.name

ec2_get_spot_price(ec2)



#List of region -----------------------------------------------
rs= ec2.get_all_regions()
for x in rs:  print x.name


#Get List of Image ------------------------------------------------------------------------------------
image_status = ec2.get_all_images(); print 'Start Printing'
for x in image_status :
   name1= '' if x.name is None else x.name
   if x.state=='available' and name1.find(('amazon')) > -1 :
      print x.id, x.name, x.state


#Get List of Security Group  ---------------------------------------------------------------------------
rs = conn.get_all_security_groups()
for x in rs:  print x.id, x.name
len(rs)



keypair,_= region_get_keypair(ec2)


req=ec2.request_spot_instances(price='0.007',
                               image_id='ami-2911c647',
                               instance_type='c4.large',
                               key_name=keypair,
                               user_data='',
                               security_group_ids= ['sg-2d656a44']
                               )[0]




#Get All instances
rs= ec2.get_all_instances()
for x in rs: print x.id





'''
def instance_wait_forfilling(conn, request_ids, pending_request_ids):
    """Loop through all pending request ids waiting for them to be fulfilled."""
    while True :
      results = conn.get_all_spot_instance_requests(request_ids=pending_request_ids)
      for result in results:
        if result.status.code == 'fulfilled':
            pending_request_ids.pop(pending_request_ids.index(result.id))
            print "aws request `{}` fulfilled!".format(result.id)
        else:
            print "waiting on `{}`".format(result.id)

      if len(pending_request_ids) == 0:
        print "all spots fulfilled!"; sys.exit(0)
      else:
        time.sleep(50)
'''

# Request a aws instance,  you'll probably want to make these requests in multiple availability zones
requests = [ec2.request_spot_instances('0.005', 'ami-983ce8f6', count=1,
            type='one-time', instance_type='t2.micro')]


request_ids = [req.id for req in request for request in requests]


# Wait for our spots to fulfill
instance_wait_forfilling(ec2, request_ids, copy.deepcopy(request_ids))


rs= ec2.get_all_spot_instance_requests()
for x in rs: print x.id


fulfilled = ec2.get_all_spot_instance_requests(filters={'status-code': 'fulfilled'})




####################################################################################################

'''

def config_read_user_data_from_local(file1):
   user_data=config.get('EC2', 'user_data')
   if config.get('EC2', 'user_data') is None or user_data=='':
      try:
         user_data=(open(config.get('EC2', 'user_data_file'), 'r')).read()
      except:
         user_data=''
   return user_data


'''


'''

class ec2_monitor(Object):
   def __init__(self):
      self.instance_active=[]
      self.instance_histo=[]  # Pandas of instance activated
      self.instance_type=[]  # (region, instanceType Mx4.5)
      self.instance_region=[]  # (region, instanceType Mx4.5)
      self.bidprice=[]  # (region, instanceType, bidprice)

      # inst= name, region, type, ami, ip,
      pass

   def instance_add_db(self, region, type1, bidprice, ami, inst_ip):
      pass

   def monitor_start(self):
      pass

   def monitor_stop(self):
      instance_stop(ec2, inst)


      elif action == 'stop' and inst is not None:
              instance_stop(ec2, inst)
      elif action == 'list':
              print 'Active Spot Instnaces (AMI: %s)' % config.get('EC2', 'ami')
              list_all_existing_instances(ec2)
      else:
              print 'No action taken'

      pass


def db_instance_add(instance, dbname='sqlite:///aapackage/aws/aws.sqlite'):
   import sys, os, time, datetime, csv, sqlalchemy as sql, pandas as pd
   os.chdir('D:/_devs/Python01/project27/');
   sys.path.append('D:/_devs/Python01/project27/aapackage/')
   import util

   db_con=sql.create_engine(db_name, execution_options={'sqlite_raw_colnames': True})

   i= instance
   allinst=[i.id, i.region, i.instance_type, i.image_id, i.dns_name, i.state, i.launch_time,   datetime.now()]
   col1=['id', 'region', 'instance_type', 'image_id', 'dns_name', 'state', 'launch_time',   'date']

   df=util.pd_array_todataframe(allinst, col1)
   df.to_sql('ec2_instance_active', db_con, flavor='sqlite', if_exists='append')
   df.to_sql('ec2_instance_histo',  db_con, flavor='sqlite', if_exists='append')


def db_instance_update(conn, dbname='sqlite:///aapackage/aws/aws.sqlite'):
   import sys, os, time, sqlalchemy as sql
   os.chdir('D:/_devs/Python01/project27/');
   sys.path.append('D:/_devs/Python01/project27/aapackage/')
   import util

   db_con=sql.create_engine(db_name, execution_options={'sqlite_raw_colnames': True})

   all_inst0=list_all_existing_instances(conn)
   allinst=[]
   for i in all_inst0 :
     allinst.append(
      [i.id, i.region, i.instance_type, i.image_id, i.dns_name, i.state, i.launch_time,   datetime.now()])
   col1=['id', 'region', 'instance_type', 'image_id', 'dns_name', 'state', 'launch_time',   'date']

   df=util.pd_array_todataframe(np.array(allinst), col1)
   df.to_sql('ec2_instance_active', db_con, flavor='sqlite', if_exists='replace')

'''


'''

def instance_wait_forup(conn, inst):
   print 'Waiting for instance to be Ready'
   while True:
      s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      if inst.ip_address is None:
         inst=instance_get_existing(conn)
      try:
         if inst.ip_address is None:
            print 'IP not assigned yet ...',
         else:
            s.connect((inst.ip_address, 22))
            s.shutdown(2)
            print 'Server is up, Server Public IP - %s' % inst.ip_address
            return inst.ip_address
            break
      except:
         print '.',
      sleep(10)


def instance_printall(running_instances):
   print 'The following running instances were found'
   for account_name in running_instances:
      print '\tAccount: %s' % account_name
      d=running_instances[account_name]
      for region_name in d:
         print '\t\tRegion: %s' % region_name
         for instance in d[region_name]:
            print '\t\t\tAn %s instance: %s' % (instance.instance_type, instance.id)
            print '\t\t\t\tTags=%s' % instance.tags


def instance_get_allactive_allregion(accounts=None, conn=None,  quiet=False):
   """Will find all running instances across all EC2 regions for all of theaccounts supplied.
   :type accounts: dict
   :param accounts: A dictionary contain account information. The key is
   a string identifying the account (e.g. "dev") and the
   value is a tuple or list containing the access key and secret key, in that order.
   If this value is None, the credentials in the boto config will be used.
   """
   if not accounts:
      creds= aws_accesskey_get()
      #creds=(boto.config.get('Credentials', 'aws_access_key_id'), boto.config.get('Credentials', 'aws_secret_access_key'))
      accounts={'main': creds}

   running_instances={}
   for account_name in accounts:
      running_instances[account_name]={}
      ak, sk=accounts[account_name]
      for region in boto.ec2.regions():
         conn=region.connect(aws_access_key_id=ak, aws_secret_access_key=sk)
         filters={'instance-state-name': 'running'}
         instances=[]
         reservations=conn.get_all_instances(filters=filters)
         for r in reservations:
            instances+=r.instances
            if instances: running_instances[account_name][region.name]=instances
            if not quiet: instance_printall(running_instances)
   return running_instances


def instance_get_allactive(conn):
   from datetime import datetime
   d1=datetime.now()
   running_instances= instance_get_allactive_allregion(accounts=None, quiet=True)
   inst_list=[]
   for account_name in running_instances:
      d=running_instances[account_name]
      for region_name in d:
         for inst in d[region_name]:
            inst_list.append(inst)
   return inst_list

'''



'''

def instance_get_existing(conn):
   instances=conn.get_all_instances() # filters={'tag:Name': config.get('EC2', 'tag')}
   if len(instances) > 0:
      return instances[0].instances[0]
   else:
      return None


def list_all_existing_instances2(conn):
   reservations=conn.get_all_instances()
   if len(reservations) > 0:
      inst_list=[]
      r_instances=[inst for resv in reservations for inst in resv.instances]
      for inst in r_instances:
         inst_list.append([inst.id, instance.inst_type, inst.state, instance.tags])
   return inst_list


def list_all_existing_instances(conn):
   reservations=conn.get_all_instances(filters={'tag:Name': config.get('EC2', 'tag')})
   if len(reservations) > 0:
      r_instances=[inst for resv in reservations for inst in resv.instances]
      for inst in r_instances:
         print "Instance Id: %s (%s)" % (inst.id, inst.state)
   return r_instances



'''

'''
def main():
   # Entry
   action='start' if len(sys.argv)==1 else sys.argv[1]
   conn=aws_conn_create()
   if conn is None:
      print 'Unable to create EC2 ec2'
      sys.exit(0)
   inst=instance_get_existing(conn)
   user_data=read_user_data_from_local_config()

   if action=='start':
      if inst is None or inst.state=='terminated':
         spot_price=instance_spot_get_pricelast(conn)
         print 'Spot price is ' + str(spot_price) + ' ...',
         if spot_price > float(config.get('EC2', 'max_bid')):
            print 'too high!'
            sys.exit(0)
         else:
            print 'below maximum bid, continuing'
            instance_spot_request_start(conn, user_data)
            inst=instance_get_existing(conn)
      wait_for_up(conn, inst)  # Start the instance
   elif action=='stop' and inst is not None:
      instance_stop(conn, inst)
   elif action=='list':
      print 'Active Spot Instnaces (AMI: %s)' % config.get('EC2', 'ami')
      list_all_existing_instances(conn)
   else:
      print 'No action taken'
'''

'''
def create_instances(module, ec2, override_count=None):
    """
    Creates new instances
    module : AnsibleModule object
    ec2: authenticated ec2 connection object
    Returns:
        A list of dictionaries with instance information
        about the instances that were launched
    """
    key_name = module.params.get('key_name')
    id = module.params.get('id')
    group_name = module.params.get('group')
    group_id = module.params.get('group_id')
    zone = module.params.get('zone')
    instance_type = module.params.get('instance_type')
    spot_price = module.params.get('spot_price')
    image = module.params.get('image')




GivenSecGroup=conn.get_all_security_groups(sec_group_id)
to
GivenSecGroup=conn.get_all_security_groups(group_ids=[sec_group_id])
Long Answer

'''


'''
   elif action == 'stop' and inst is not None:
           instance_stop(ec2, inst)
   elif action == 'list':
           print 'Active Spot Instnaces (AMI: %s)' % config.get('EC2', 'ami')
           list_all_existing_instances(ec2)
   else:
           print 'No action taken'

    By Script can delete Instance:

     launch_listener_ec2_spot()   in Sub-Process

     Class containing item

'''

'''
http://boto.cloudhackers.com/en/latest/ref/ec2.html
request_spot_instances(price, image_id, count=1, type='one-time', valid_from=None, valid_until=None, launch_group=None, availability_zone_group=None, key_name=None, security_groups=None, user_data=None, addressing_type=None, instance_type='m1.small', placement=None, kernel_id=None, ramdisk_id=None, monitoring_enabled=False, subnet_id=None, placement_group=None, block_device_map=None, instance_profile_arn=None, instance_profile_name=None, security_group_ids=None, ebs_optimized=False, network_interfaces=None, dry_run=False)
Request instances on the aws market at a particular price.

Parameters:
price (str) – The maximum price of your bid
image_id (string) – The ID of the image to run
count (int) – The of instances to requested
type (str) – Type of request. Can be ‘one-time’ or ‘persistent’. Default is one-time.
valid_from (str) – Start date of the request. An ISO8601 time string.
valid_until (str) – End date of the request. An ISO8601 time string.
launch_group (str) – If supplied, all requests will be fulfilled as a group.
availability_zone_group (str) – If supplied, all requests will be fulfilled within a single availability zone.
key_name (string) – The name of the key pair with which to launch instances
security_groups (list of strings) – The names of the security groups with which to associate instances
user_data (string) – The user data passed to the launched instances
instance_type (string)

'''


'''
from pprint import pprint
from boto import ec2

AWS_ACCESS_KEY_ID = 'XXXXXXXXXXXXXXXXXX'
AWS_SECRET_ACCESS_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

ec2conn = ec2.connection.EC2Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
reservations = ec2conn.get_all_instances()
instances = [i for r in reservations for i in r.instances]
for i in instances:
    pprint(i.__dict__)
    break # remove this to list all instances
Results:

{'_in_monitoring_element': False,
 'ami_launch_index': u'0',
 'architecture': u'x86_64',
 'block_device_mapping': {},
 'connection': EC2Connection:ec2.amazonaws.com,
 'dns_name': u'ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com',
 'id': u'i-xxxxxxxx',
 'image_id': u'ami-xxxxxxxx',
 'instanceState': u'\n                    ',
 'instance_class': None,
 'instance_type': u'm1.large',
 'ip_address': u'xxx.xxx.xxx.xxx',
 'item': u'\n                ',
 'kernel': None,
 'key_name': u'FARM-xxxx',
 'launch_time': u'2009-10-27T17:10:22.000Z',
 'monitored': False,
 'monitoring': u'\n                    ',
 'persistent': False,
 'placement': u'us-east-1d',
 'previous_state': None,
 'private_dns_name': u'ip-10-xxx-xxx-xxx.ec2.internal',
 'private_ip_address': u'10.xxx.xxx.xxx',
 'product_codes': [],
 'public_dns_name': u'ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com',
 'ramdisk': None,
 'reason': '',
 'region': RegionInfo:us-east-1,
 'requester_id': None,
 'rootDeviceType': u'instance-store',
 'root_device_name': None,
 'shutdown_state': None,
 'spot_instance_request_id': None,
 'state': u'running',
 'state_code': 16,
 'subnet_id': None,
 'vpc_id': None}

'''









'''
Expected behavior

python main.py
Spot price is 0.02... below maximum bid, continuing
Spot request created, status: open
Waiting for instance to be provisioned (usually takes 1m to be reviewed, another 2m to be fulfilled) ...  . . . Instance is active.
Tagging instance... Done.
Waiting for server to come up ... . . . . Server is up!


Subsequent connects:

python main.py
Instance exists already, we will not be provisioning another one
Waiting for server to come up ... Server is up!


Terminating (and detagging) the instance:

python main.py stop
Terminating i-0a1b1930 ... done.

'''
