
dirn="$(/home/ubuntu/)"
which_python="$(which python)"

#  find /full/path/to/dir -name "*c*" -type dir
#  python  batch/ztest/tasks/zztsk_fast/main.py
#  rm -rf ${dirn}/ztest/tasks/task_demo1


#rm -rf `find  -type d -name "*task_demoxx1*"`
#yes | cp -rf ${dirn}/ztest/tasks/task_ignore  ${dirn}/../../ztest/tasks/task_demoxx1

#rm -rf `find  -type d -name "*task_demoxx2*"`
#yes | cp -rf ${dirn}/ztest/tasks/task_ignore_pygmo  ${dirn}/../../ztest/tasks/task_demoxx2


##########################
#  parser.add_argument("--task_folder", default=TASK_FOLDER_DEFAULT,  help="Absolute or relative path to the working folder.")
#  parser.add_argument("--log_file", default="logfile_batchdaemon.log", help=".")
#  parser.add_argument("--mode", default="nodaemon", help="daemon/ .")
#  parser.add_argument("--waitsec", type=int, default=30, help="wait sec")



##########################
#    parser = argparse.ArgumentParser(  description='Record CPU and memory usage for a process')
#    parser.add_argument('--monitor_log_file', type=str, default=MONITOR_LOG_FILE,  help='output the statistics ')
#    parser.add_argument('--duration',         type=float,  help='how long to record in secs.')
#    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,  help='wait tine in secs.')
#    parser.add_argument('--monitor_log_folder', type=str, default=MONITOR_LOG_FOLDER,  help='')
#    parser.add_argument('--log_file', type=str, default="log_batchdaemon_monitor.log",help='daemon log')
#    parser.add_argument("--mode", default="nodaemon", help="daemon/ .")



################################################################################
###### Run Batch  ##############################################################
source activate py36batch

batch_daemon_launch_cli.py  --task_folder  ${dirn}/tasks  --log_file ${dirn}/zlog/batchdaemong.log  --mode daemon  --waitsec 60

sleep 3

batch_daemon_monitor_cli.py --monitor_log_folder ${dirn}/tasks_out/   --monitor_log_file monitor_log_file.log   --log_file ${dirn}/zlog/batchdaemon_monitor.log    --mode daemon 





