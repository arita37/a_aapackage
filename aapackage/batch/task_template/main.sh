#!/bin/bash

## /home/ubuntu/zbatch.sh
### nohup  /home/ubuntu/zbatch.sh  2>&1 | tee -a /home/ubuntu/zlog/zbatch_log.log
## /home/ubuntu/aagit/aapackage/aapackage/batch/task_template/main.sh


### Need to source when using SSH   ###########################################
echo "-Start Script-"

source /home/ubuntu/.bashrc
export PATH="/home/ubuntu/anaconda3/bin:$PATH"

source activate base
# cd /home/ubuntu/
# whoami
pwd
which python
which conda



########Change to Script Folder ##############################################
CUR_DIR=$(dirname "${0}")
# echo "${CUR_DIR}"
cd $CUR_DIR
pwd


##############################################################################
###### Task Launcher  ########################################################
# conda env create -n myenv -f requirement.yml
# source activate myenv

source activate py36
which python



python ./main_run.py




##############################################################################










