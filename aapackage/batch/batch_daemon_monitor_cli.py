# -*- coding: utf-8 -*-
'''Daemon monitoring batch '''
import os, sys
import argparse
import logging
from time import sleep


import arrow
import psutil

###############################################################################
from aapackage import util_log
from aapackage.batch import util_batch
from aapackage.batch import util_cpu




############### Variable definition ###########################################
WORKING_FOLDER = os.path.dirname(__file__)
MONITOR_LOG_FOLDER = os.path.join(WORKING_FOLDER, "ztest", "monitor_logs")
MONITOR_LOG_FILE = MONITOR_LOG_FOLDER + "/" + "batch_monitor_" + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss") + ".log"


DEFAULT_INTERVAL = 30  # seconds
DEFAULT_DURATION = 3600  # seconds
PYTHON_COMMAND   = str(sys.executable)
PROCESS_TO_LOOK = "python"
logger = logging.basicConfig()


################################################################################
def log(*argv):
    logger.info(",".join([str(x) for x in argv]))



###############################################################################
def load_arguments():
    parser = argparse.ArgumentParser(
        description='Record CPU and memory usage for a process')

    parser.add_argument('--monitor_log_file', type=str, default=MONITOR_LOG_FILE,  help='output the statistics ')
    parser.add_argument('--duration', type=float,  help='how long to record in secs.')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,  help='wait tine in secs.')
    parser.add_argument('--monitor_log_folder', type=str, default=MONITOR_LOG_FOLDER,  help='')
    parser.add_argument('--log_file', type=str, default="batch/ztest/logfile_batchdaemon_monitor.log",help='daemon log')
    args = parser.parse_args()
    return args


###############################################################################
if __name__ == '__main__':
    args = load_arguments()
    logger = util_log.logger_setup(__name__,
                                   log_file=args.log_file,
                                   formatter=util_log.FORMATTER_4)

    log("batch_daemon_monitor_cli.py;Process started.")
    util_batch.os_folder_create(folder= args.monitor_log_folder)

    batch_root_pid = util_cpu.ps_find_procs_by_name( name="python",
                                                     cmdline="batch_daemon_launch_cli.py" )
    if len(batch_root_pid) > 0:
        util_cpu.ps_process_monitor_child(batch_root_pid[0],
                                          logfile  = args.monitor_log_file,
                                          duration = args.duration, interval=args.interval)
    log("batch_daemon_monitor_cli.py;Process Completed.")







