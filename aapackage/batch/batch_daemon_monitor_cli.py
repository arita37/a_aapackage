# -*- coding: utf-8 -*-
"""Daemon monitoring batch

  ###  monitor subprocess launched in specific folder
       when task is finished, rename folder to _qdone

  ### Folder pattern where process are searched :
    #  /tasks/t53453/main.py   or /tasks/t53453/main.sh  

  ### CLI                                     
batch_daemon_monitor_cli.py --monitor_log_folder   tasks_out/   --monitor_log_file monitor_log_file.log   --log_file   zlog/batchdaemon_monitor.log    --mode daemon 


Folder suffix rules :
 _qstart :  task running
 _qdone  :  task finish
 _ignore :  folder to be ignored
 
 
"""
import argparse
import logging
import os
import sys
from time import sleep

import arrow
import psutil

####################################################################################################
from aapackage import util_log
from aapackage.batch import util_batch, util_cpu

############### Variable definition ################################################################
# MONITOR_LOG_FILE = MONITOR_LOG_FOLDER + "/" + "batch_monitor_" + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss") + ".log"
CWD_FOLDER = os.getcwd()


####################################################################################################
logger = logging.basicConfig()


def log(*argv):
    logger.info(",".join([str(x) for x in argv]))


def logcpu(*argv):
    loggercpu.info(",".join([str(x) for x in argv]))


####################################################################################################
def load_arguments():
    parser = argparse.ArgumentParser(description="Monitor python process from tasks")
    parser.add_argument("--verbose", default=0, help="verbose")
    parser.add_argument(
        "--log_file",
        type=str,
        default=CWD_FOLDER + "log_batchdaemon_monitor.log",
        help="daemon log",
    )
    parser.add_argument("--mode", default="nodaemon", help="daemon/ .")

    parser.add_argument(
        "--monitor_log_file",
        type=str,
        default=CWD_FOLDER + "log_batchdaemon_cpu.log",
        help="output the statistics ",
    )
    parser.add_argument(
        "--process_folder", default="/home/ubuntu/tasks/", help="process name pattern"
    )
    parser.add_argument("--process_isregex", default=1, help="process name pattern regex")
    parser.add_argument("--waitsec", type=int, default=10, help="sleep")

    args = parser.parse_args()
    return args


####################################################################################################
if __name__ == "__main__":
    args = load_arguments()
    logger = util_log.logger_setup(
        __name__, log_file=args.log_file, formatter=util_log.FORMATTER_4, isrotate=True
    )

    ### Process CPU usage
    loggercpu = util_log.logger_setup(
        __name__ + "logcpu",
        log_file=args.monitor_log_file,
        formatter=util_log.FORMATTER_4,
        isrotate=True,
    )

    batch_pid_dict = {}
    p_pattern = args.process_folder
    p_pattern = p_pattern[:-1] if p_pattern.endswith("/") else p_pattern

    #  /tasks/t53453/main.py   or /tasks/t53453/main.sh
    regex_pattern = r"((.*/)?%s/t.*/main\.(py|sh))" % p_pattern

    log("Monitor started, regex: ", regex_pattern)
    logcpu("Process monitor started", "", "")
    while True:
        batch_pid = util_cpu.ps_find_procs_by_name(
            name=regex_pattern, ishow=args.verbose, isregex=args.process_isregex
        )

        # Add new PID
        for pid in batch_pid:
            if pid["pid"] not in batch_pid_dict and len(pid["cmdline"]) > 0:
                log("PID added", pid)
                batch_pid_dict[pid["pid"]] = pid
                logcpu("Added", pid["pid"], pid["cmdline"])

        log("PID", batch_pid_dict)

        ddict = {k: v for k, v in batch_pid_dict.items() if v}
        for pid, pr in ddict.items():
            """
         try : 
           status = util_cpu.ps_get_process_status(psutil.Process(pid))
           log(pid, status )         
         except Exception as e :
             log(e)
         """
            try:
                flag = util_cpu.ps_process_isdead(pid)
                if flag:
                    log("Dead", pr)
                    logcpu("Dead", pr["pid"], pr["cmdline"])

                    os.system("pkill -9 " + str(pid))
                    path = pr["cmdline"][1]
                    del batch_pid_dict[pid]

                    path = os.path.dirname(os.path.abspath(path))
                    util_batch.os_folder_rename(path, path.replace("_qstart", "_qdone"))
                    log("_qdone", path)

            except Exception as e:
                log(e)

        if args.mode != "daemon":
            log("Monitor daemon exited")
            break
        sleep(args.waitsec)


"""

if __name__ == '__main__':
    args   = load_arguments()
    logger = util_log.logger_setup(__name__,
                                   log_file  = args.log_file,
                                   formatter = util_log.FORMATTER_4)

    util_batch.os_folder_create(folder= args.monitor_log_folder)

    while True :
      log("Monitor started.")    
      batch_root_pid = util_cpu.ps_find_procs_by_name( name="python",
                                                       cmdline= args.process_pattern )

      for pid in batch_root_pid :
         util_cpu.ps_process_monitor_child( pid,
                                            logfile  = args.monitor_log_file,
                                            duration = args.duration, interval = args.interval)

      log("Monitor Completed.")
      if args.mode != "daemon" : 
        log("Monitor daemon exited")  
        break
      sleep(10)

"""
