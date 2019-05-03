#!/bin/bash

## /home/ubuntu/zbatch.sh
### nohup  /home/ubuntu/zbatch.sh  2>&1 | tee -a /home/ubuntu/zlog/zbatch_log.log

### Need to source when using SSH   ############################################
echo "-Start Script-"
source /home/ubuntu/.bashrc


export PATH="/home/ubuntu/anaconda3/bin:$PATH"

source activate base
cd /home/ubuntu/
whoami
pwd
which python
which conda


##############################################################################
###### Task Launcher  ########################################################
source activate py36
which python

python ./main_run.py


##############################################################################










