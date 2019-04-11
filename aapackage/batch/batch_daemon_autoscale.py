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
         
  

ss  = 'aws ec2 request-spot-instances   --region us-west-2  --spot-price "0.55" --instance-count 1 '
ss += ' --type "one-time" --launch-specification "file://D:\_devs\Python01\\awsdoc\\ec_config2.json" '


util_aws
  



'''
import os,sys

from aapackage import util_aws




