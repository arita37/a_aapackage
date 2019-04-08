# -*- coding: utf-8 -*-
'''Daemon monitoring batch

  monitor subprocess launched by batch_daemon_laucnh_cli.py
  when task is finished, rename folder to _qdone

  util_cpu.ps_find_procs_by_name( name="python", cmdline="tasks/" )
                                     
                                     
batch_daemon_monitor_cli.py --monitor_log_folder   tasks_out/   --monitor_log_file monitor_log_file.log   --log_file   zlog/batchdaemon_monitor.log    --mode daemon 
 _qstart
 _qdone
 
 

'''
import os, sys
import argparse
import logging
from time import sleep


import arrow
import psutil

####################################################################################################
from aapackage import util_log
from aapackage.batch import util_batch
from aapackage.batch import util_cpu




############### Variable definition ################################################################
WORKING_FOLDER = os.path.dirname(__file__)
MONITOR_LOG_FOLDER = os.path.join(WORKING_FOLDER, "ztest", "monitor_logs")
MONITOR_LOG_FILE = MONITOR_LOG_FOLDER + "/" + "batch_monitor_" + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss") + ".log"


# DEFAULT_INTERVAL = 30  # seconds
# DEFAULT_DURATION = 3600  # seconds
# PYTHON_COMMAND   = str(sys.executable)
# PROCESS_TO_LOOK  = "python"


####################################################################################################
logger = logging.basicConfig()
def log(*argv):
    logger.info(",".join([str(x) for x in argv]))



def logcpu(*argv):
    loggercpu.info(",".join([str(x) for x in argv]))



####################################################################################################
def load_arguments():
    parser = argparse.ArgumentParser(  description='Record CPU and memory usage for a process')
    parser.add_argument('--monitor_log_file', type=str, default=MONITOR_LOG_FILE,  help='output the statistics ')
    # parser.add_argument('--duration',         type=float,  help='how long to record in secs.')
    # parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,  help='wait tine in secs.')
    # parser.add_argument('--monitor_log_folder', type=str, default=MONITOR_LOG_FOLDER,  help='')
    parser.add_argument('--log_file', type=str, default="log_batchdaemon_monitor.log",help='daemon log')
    parser.add_argument("--mode", default="nodaemon", help="daemon/ .")
    parser.add_argument("--process_pattern", default="tasks/", help="process name pattern")
    parser.add_argument("--waitsec", type=int, default=10, help="sleep")
    
    args = parser.parse_args()
    return args


####################################################################################################
if __name__ == '__main__':
    args   = load_arguments()
    logger = util_log.logger_setup(__name__,
                                   log_file  = args.log_file,
                                   formatter = util_log.FORMATTER_4,
                                   isrotate=True)

    ### Process
    loggercpu = util_log.logger_setup(__name__ + "logcpu",
                                   log_file  = args.monitor_log_file,
                                   formatter = util_log.FORMATTER_4,
                                   isrotate=True)

                                 
    # util_batch.os_folder_create(folder= args.monitor_log_folder)



    batch_pid_dict = {}
    log("Monitor started.")   
    logcpu("Process monitor started", "", "" )
    while True :
      batch_pid = util_cpu.ps_find_procs_by_name( name= "python", ishow=0,
                                                  cmdline= args.process_pattern )

      # Add new PID
      for pid in batch_pid :
         if pid["pid"] not in batch_pid_dict and len( pid["cmdline"] ) > 0 :
            log("PID added", pid) 
            batch_pid_dict[pid["pid"] ] = pid
            logcpu( "Added",  pid["pid"], pid["cmdline"]  )
            
            
      log("PID", batch_pid_dict)

      ddict = {k: v for k,v in batch_pid_dict.items() if v }
      for pid, pr in ddict.items() :
         """
         try : 
           status = util_cpu.ps_get_process_status(psutil.Process(pid))
           log(pid, status )         
         except Exception as e :
             log(e)
         """
         try :
           flag = util_cpu.ps_process_isdead( pid )
           if flag :
             log("Dead", pr)
             logcpu( "Dead",  pr["pid"], pr["cmdline"]  )
             
             os.system("pkill -9 " + str(pid) )
             path = pr["cmdline"][1]
             del batch_pid_dict[ pid ] 
             
             
             
             path = os.path.dirname(os.path.abspath(path))  
             util_batch.os_folder_rename( path ,  path.replace("_qstart", "_qdone") )
             log("_qdone", path)    

         except Exception as e :
            log(e)
        
             
      if args.mode != "daemon" : 
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



