# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

import util

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









'''
Compute time :
Start : 4pm 16  End: 6pm07
2hours : 96 * 100*15*8*25,   28800000
15000 / min / CPU

'''



'''
Check if ipython process is running

host=   '52.79.79.1'
ssh= util.aws_ec2_ssh(host)

aux= ssh.cmd2('pgrep ipython')
#  Out[8]: [('', '')]
[('1671\n', '')]


  if aux.find('ipython') == -1 : ipython_idle_count+= 1
  else :                         ipython_idle_count= 0

  if ipython_idle_count > 2 :
       break; sys.exit(0)



pgrep -f php5
Unlike the ps | grep construction with which you need to filter out the grep line or use pattern tricks, pgrep just won't pick itself by design.
If you want full details instead of just the pids, you can use:
ps wup $(pgrep -f python)


'''

'''' EC2
https://blog.cloudability.com/aws-data-transfer-costs-what-they-are-and-how-to-minimize-them/


1) Instance Cost
1) Data Transfer Cost to Internet:
        0.1 Go * 0.09  USD     ( 0.09 USD / GB)

3) EBS storage Cost :
     5 hour * 0.12 USD /24.0 hour /30.0 day

     $0.114 per GB-month of provisioned storage    SSD

     $0.05 per GB-month of data stored    Amazon EBS Snapshots to Amazon S3

    ($0.10 per GB-month * 2000 GB * 12 hours / (24 hours/day * 30 day-month)).



Announcing Amazon S3 Reduced Redundancy Storage




'''
