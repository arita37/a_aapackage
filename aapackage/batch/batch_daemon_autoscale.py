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


{
  "ImageId": "ami-0b01d9c82fc6d1391",
  "KeyName": "aws_ec2_4", 
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
import os,sys

from aapackage import util_aws




