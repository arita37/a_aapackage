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
         
  





'''