
dirn="$(dirname "$0")"
which_python="$(which python)"

#  find /full/path/to/dir -name "*c*" -type dir
#  python  batch/ztest/tasks/zztsk_fast/main.py
#  rm -rf ${dirn}/ztest/tasks/task_demo1


#rm -rf `find  -type d -name "*task_demoxx1*"`
#yes | cp -rf ${dirn}/ztest/tasks/task_ignore  ${dirn}/../../ztest/tasks/task_demoxx1

#rm -rf `find  -type d -name "*task_demoxx2*"`
#yes | cp -rf ${dirn}/ztest/tasks/task_ignore_pygmo  ${dirn}/../../ztest/tasks/task_demoxx2



###### Run Batch  ##############################################################
batch_daemon_launch_cli.py  --task_folder ${dirn}/tasks/ &
sleep 3
batch_daemon_monitor_cli.py --monitor_log_folder ${dirn}/log



