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
MONITOR_LOGS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "monitor_logs")
DEFAULT_LOG_FILE = os.path.join(MONITOR_LOGS_DIRECTORY, arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
                                + "_batch_monitor.log")
DEFAULT_INTERVAL = 10  # seconds
DEFAULT_DURATION = 3600  # seconds
PYTHON_COMMAND = "python"

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
    util_log.printlog(s1=message)


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


if __name__ == '__main__':

    args = load_arguments()
    if not os.path.isdir(MONITOR_LOGS_DIRECTORY):
        os.mkdir(MONITOR_LOGS_DIRECTORY)
    log_file = args.log
    duration = args.duration
    interval = args.interval
    error_log = open(WORKING_DIRECTORY+"/errors_daemon_monitor.log", "a")
    pid_of = "%s batch_daemon_launch_cli.py" % PYTHON_COMMAND
    proc = subprocess.Popen(['pidof %s' % pid_of], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    if err:
        log_message("batch_daemon_launch_cli.py;get_pid_by_command;command=pidof ;error: %s" % err)
    else:
        parent_process_id = out.split(" ")
        required_pid = None
        for each_pid in parent_process_id:
            try:
                pid = int(re.sub('[^0-9]', '', each_pid))
                pr = psutil.Process(pid)
                if "batch_daemon_launch_cli.py" in pr.cmdline():
                    required_pid = pid
                    break
            except Exception as ex:
                pass
        if required_pid:
            monitor(required_pid, logfile=log_file, duration=duration, interval=interval)