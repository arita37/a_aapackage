# -*- coding: utf-8 -*-
'''Daemon monitoring batch '''
import os, sys
import numpy as np

import argparse
import gc
import logging
import socket
from time import sleep
import time
import ujson
import arrow
import re
import subprocess
import psutil
import util_log
###############################################################################


############### Variable definition ###########################################
logging.basicConfig(level=logging.INFO)
APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MONITOR_LOGS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "ztest", "monitor_logs")
DEFAULT_LOG_FILE = os.path.join(MONITOR_LOGS_DIRECTORY, arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
                                + "_batch_monitor.log")
DEFAULT_INTERVAL = 10  # seconds
DEFAULT_DURATION = 3600  # seconds
PYTHON_COMMAND = str(sys.executable)
PROCESS_TO_LOOK = "python"
#########Logging ##############################################################


def load_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Record CPU and memory usage for a process')

    parser.add_argument('--log', type=str, default=DEFAULT_LOG_FILE,
                        help='output the statistics to a file')

    parser.add_argument('--duration', type=float,
                        help='how long to record for (in seconds). If not '
                             'specified, the recording is continuous until '
                             'the job exits.')

    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help='how long to wait between each sample (in '
                             'seconds). By default the process is sampled '
                             'as often as possible.')
    parser.add_argument('--log_directory', type=str, default="",
                        help='')

    args = parser.parse_args()
    return args



def get_percent(process):
    try:
        return process.cpu_percent()
    except AttributeError:
        return process.get_cpu_percent()



def get_memory(process):
    try:
        return process.memory_info()
    except AttributeError:
        return process.get_memory_info()



def all_children(pr):
    processes = []
    children = []
    try:
        children = pr.children()
    except AttributeError:
        children = pr.get_children()
    except Exception:  # pragma: no cover
        pass

    for child in children:
        processes.append(child)
        processes += all_children(child)
    return processes


def log_message(message):
    util_log.printlog(app_id=APP_ID, s1=message)


def monitor(pid, logfile=None, duration=None, interval=None):
    # We import psutil here so that the module can be imported even if psutil
    # is not present (for example if accessing the version)
    log_message("Monitoring Started for Process Id: %s" % str(pid))
    pr = psutil.Process(pid)

    # Record start time
    start_time = time.time()

    if logfile:
        f = open(logfile, 'w')
        f.write("# {0:12s} {1:12s} {2:12s} {3:12s} {4:12s}\n".format(
            'Timestamp'.center(12),
            'Elapsed time'.center(12),
            'CPU (%)'.center(12),
            'Real (MB)'.center(12),
            'Virtual (MB)'.center(12))
        )

    try:

        # Start main event loop
        while True:

            # Find current time
            current_time = time.time()

            try:
                pr_status = pr.status()
            except TypeError:  # psutil < 2.0
                pr_status = pr.status
            except psutil.NoSuchProcess:  # pragma: no cover
                break

            # Check if process status indicates we should exit
            if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                log_message("Process finished ({0:.2f} seconds)"
                      .format(current_time - start_time))
                break

            # Check if we have reached the maximum time
            if duration is not None and current_time - start_time > duration:
                break

            # Get current CPU and memory
            try:
                current_cpu = get_percent(pr)
                current_mem = get_memory(pr)
            except Exception:
                break
            current_mem_real = current_mem.rss / 1024. ** 2
            current_mem_virtual = current_mem.vms / 1024. ** 2

            # Get information for children
            for child in all_children(pr):
                try:
                    current_cpu += get_percent(child)
                    current_mem = get_memory(child)
                except Exception:
                    continue
                current_mem_real += current_mem.rss / 1024. ** 2
                current_mem_virtual += current_mem.vms / 1024. ** 2

            if logfile:
                timestamp = str(arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,"))
                f.write("{0:12} {1:12.3f} {2:12.3f} {3:12.3f} {4:12.3f}\n".format(
                    timestamp,
                    current_time - start_time,
                    current_cpu,
                    current_mem_real,
                    current_mem_virtual))
                f.flush()

            sleep(interval)

    except KeyboardInterrupt:  # pragma: no cover
        pass

    if logfile:
        f.close()



def ps_find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    assert name, name
    ls = []
    for p in psutil.process_iter():
        name_, exe, cmdline = "", "", []
        try:
            name_ = p.name()
            cmdline = p.cmdline()
            exe = p.exe()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except psutil.NoSuchProcess:
            continue
        if name == name_ or cmdline[0] == name or os.path.basename(exe) == name:
            ls.append(name)
    return ls


if __name__ == '__main__':
    util_log.LOG_FILE = "ztest/logfile_batchdaemon.log"
    log_message("batch_daemon_monitor_cli.py;Process started.")
    args = load_arguments()
    log_file = args.log
    duration = args.duration
    interval = args.interval
    log_directory = args.log_directory

    if not os.path.isdir(MONITOR_LOGS_DIRECTORY):
        os.mkdir(MONITOR_LOGS_DIRECTORY)


    # Use psutil find_procs_by_name

    #error_log = open(WORKING_DIRECTORY+"/errors_daemon_monitor.log", "a")
    proc = subprocess.Popen(['pidof', PROCESS_TO_LOOK], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    out, err = proc.communicate()
    if err:
        log_message("batch_daemon_launch_cli.py;get_pid_by_command;command=pidof ;error: %s" % err)
    else:
        parent_process_id = out.split(" ")
        required_pid = None
        for each_pid in parent_process_id:
            if each_pid.strip():
                try:
                    pid = int(re.sub('[^0-9]', '', each_pid))
                    pr = psutil.Process(pid)
                    if "batch_daemon_launch_cli.py" in " ".join(pr.cmdline()):
                        required_pid = pid
                        break
                except Exception as ex:
                    log_message("batch_daemon_monitor_cli.py;exception: {0}".format(ex.message))
        if required_pid:
            monitor(required_pid, logfile=log_file, duration=duration, interval=interval)
    # error_log.close()
    log_message("batch_daemon_monitor_cli.py;Process Completed.")
