# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0601,E1123,W0614,F0401,E1120,E1101,E0611,W0702
'''
Launch processors and monitor the CPU, memory usage.
Maintain same leve of processors over time.

'''

import argparse
import copy
import csv
import logging
import os
import random
import shlex
import subprocess
import sys
from time import sleep, time

import arrow
import psutil


############# Root folder #####################################################
def os_getparent(dir0):
    return os.path.abspath(os.path.join(dir0, os.pardir))


try:
     DIRCWD = os_getparent(os.path.dirname(os.path.abspath(__file__)))

except:
    try:
        DIRCWD = os_getparent(os.path.dirname(os.path.abspath(__file__)))
        if sys.argv[0] == '':
            raise Exception
        DIRCWD = os_getparent(os.path.abspath(os.path.dirname(sys.argv[0])))
    except:
        DIRCWD = '/home/run/repo/'

############# Arg parsing #####################################################
try:
    ppa = argparse.ArgumentParser()
    ppa.add_argument('--DIRCWD', type=str, default='', help=' Root Folder')
    ppa.add_argument('--do', type=str, default='zdoc', help='action')
    ppa.add_argument('--verbose', type=int, default=0, help=' Verbose mode')
    ppa.add_argument('--test', type=int, default=0, help=' ')

    ppa.add_argument('--configfile', type=str, default='/config/config.txt',
                     help=' config file')
    arg = ppa.parse_args()
    if arg.DIRCWD != '':
        DIRCWD = arg.DIRCWD

except Exception as e:
    print(e)
    sys.exit(1)
os.chdir(DIRCWD)
sys.path.append(DIRCWD + '/aapackage')
print('Root Folder', DIRCWD)
import util

###############################################################################
###############################################################################


###############################################################################
#############Variable #########################################################
global CMDS, net_avg
APP_ID = __file__ + ',' + str(os.getpid()) + '_' + str(random.randrange(10000))
logfolder = '_'.join(
    [arg.logfolder, arg.name, arg.consumergroup, arg.input_topic,
     arrow.utcnow().to('Japan').format("YYYYMMDD_HHmm_ss"),
     str(random.randrange(1000))])
util.os_folder_create(logfolder)
LOGFILE = logfolder + '/stream_monitor_cli.txt'
Mb = 1024 * 1024
net_avg = 0.0

###############################################################################
######### Logging #############################################################
logging.basicConfig(level=logging.INFO)


def printlog(s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='',
             s9='', s10=''):
    try:
        prefix = APP_ID + ',' + arrow.utcnow().to('Japan').format(
            "YYYYMMDD_HHmmss,") + ',' + arg.input_topic
        s = ','.join(
            [prefix, str(s), str(s1), str(s2), str(s3), str(s4), str(s5),
             str(s6), str(s7), str(s8), str(s9), str(s10)])

        logging.info(s)

    except Exception as e:
        logging.info(str(e))


###############################################################################



###############################################################################
########### Utilities #########################################################
def find_procs_by_name(name, ishow=1, type1='cmdline'):
    "Return a list of processes matching 'name'."
    ls = []
    for p in psutil.process_iter(attrs=['pid', "name", "exe", "cmdline"]):
        if name in p.info['name'] or name in ' '.join(p.info['cmdline']):
            ls.append(copy.deepcopy(p))
            if ishow == 1:
                printlog(p.pid, ' '.join(p.info['cmdline']))
    return ls



def launch(commands):
    processes = []
    for cmd in commands:
        try:
            p = subprocess.Popen(cmd, shell=False)
            processes.append(p.pid)
            printlog('Launched: ', p.pid, ' '.join(cmd))
            sleep(1)

        except Exception as e:
            printlog(e)
    return processes

def terminate(processes):
    for p in processes:
        pidi = p.pid
        try:
            os.kill(p.pid, 9)
            printlog('killed ', pidi)
        except Exception as e:
            printlog(e)
            try:
                os.kill(pidi, 9)
                printlog('killed ', pidi)
            except:
                pass


def extract_commands(csv_file, has_header=False):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file, skipinitialspace=True)
        if has_header:
            headers = next(reader)  # pass header
        commands = [row for row in reader]

    return commands


def is_issue(p):
    pdict = p.as_dict()
    pidi = p.pid

    printlog('Worker PID;CPU;RAM:', pidi, pdict['cpu_percent'],
             pdict['memory_full_info'][0] / Mb)

    try:
        if not psutil.pid_exists(pidi):
            printlog('Process has been killed ', pidi)
            return True

        elif pdict['status'] == 'zombie':
            printlog('Process  zombie ', pidi)
            return True

        elif pdict['memory_full_info'][0] >= pars['max_memory']:
            printlog('Process  max memory ', pidi)
            return True

        elif pdict['cpu_percent'] >= pars['max_cpu']:
            printlog('Process MAX CPU ', pidi)
            return True

        else:
            return False
    except Exception as e:
        printlog(e)
        return True


def ps_net_send(tperiod=5):
    x0 = psutil.net_io_counters(pernic=False).bytes_sent
    t0 = time()
    sleep(tperiod)
    t1 = time()
    x1 = psutil.net_io_counters(pernic=False).bytes_sent
    return (x1 - x0) / (t1 - t0)


def is_issue_system():
    global net_avg
    try:
        if psutil.cpu_percent(interval=5) > pars['cpu_usage_total']:
            return True

        elif psutil.virtual_memory().available < pars['mem_available_total']:
            return True

        else:
            return False

    except:
        return True


def monitor():
    '''
       Launch processors and monitor the CPU, memory usage.
       Maintain same leve of processors over time.
    '''
    printlog('start monitoring', len(CMDS))
    cmds2 = []
    for cmd in CMDS:
        ss = shlex.split(cmd)
        cmds2.append(ss)

    processes = launch(cmds2)
    try:
        while True:
            has_issue = []
            ok_process = []
            printlog('N_process', len(processes))

            ### check global system  ##########################################
            if len(processes) == 0 or is_issue_system():
                printlog('Reset all process')
                lpp = find_procs_by_name(pars['proc_name'], 1)
                terminate(lpp)
                processes = launch(cmds2)
                sleep(5)

            ## pid in process   ###############################################
            for pidi in processes:
                try:
                    p = psutil.Process(pidi)
                    printlog('Checking', p.pid)

                    if is_issue(p):
                        has_issue.append(p)

                    else:
                        printlog('Process Fine ', pidi)
                        ok_process.append(p)

                except Exception as e:
                    printlog(e)

            ### Process with issues    ########################################
            for p in has_issue:
                try:
                    printlog('Relaunching', p.pid)
                    pcmdline = p.cmdline()
                    pidlist = launch(
                        [pcmdline])  # New process can start before

                    sleep(3)
                    terminate([p])
                except:
                    pass

            ##### Check the number of  processes    ###########################
            sleep(5)
            lpp = find_procs_by_name(pars['proc_name'], 1)

            printlog('Active process', len(lpp))
            if len(lpp) < pars['nproc']:
                for i in range(0, pars['nproc'] - len(lpp)):
                    pidlist = launch([shlex.split(pars['proc_cmd'])])

            else:
                for i in range(0, len(lpp) - pars['nproc']):
                    pidlist = terminate([lpp[i]])

            sleep(5)
            lpp = find_procs_by_name(pars['proc_name'], 0)
            processes = [x.pid for x in lpp]

            printlog('Waiting....')
            sleep(arg.nfreq)

    except Exception as e:
        printlog(e)


if __name__ == '__main__':
    ################## Initialization #########################################
    printlog(' Initialize workers', arg.name)

    if arg.name == 'stream_couchbase':
        pars = {'max_memory'         : 1500.0 * Mb, 'max_cpu': 85.0,
                'proc_name'          : 'streaming_couchbase_update_cli.py',
                'nproc'              : arg.nproc,
                'proc_cmd'           : 'python kafkastreaming/streaming_couchbase_update_cli.py   --consumergroup {0} --nlogfreq {1}  --logfile {2} --verbose {3}  --input_topic {4}  --test {5}  --mode {6} '.format(
                    arg.consumergroup + 'couch' + arg.mode, arg.nlogfreq,
                    logfolder + '/stream_couchbase_' + str(
                        arg.consumergroup) + '.txt', arg.verbose,
                    arg.input_topic, arg.test, arg.mode),
                'mem_available_total': 2000.0 * Mb, 
                'cpu_usage_total': 98.0}
        CMDS = [pars['proc_cmd']] * pars['nproc']

    elif arg.name == 'test':
        pars = {'max_memory'                 : 2000.0 * Mb, 'max_cpu': 95.0,
                'proc_name'                  : 'streaming_test_cli.py',
                'proc_cmd'                   : 'python kafkastreaming/streaming_test_run.py    --consumergroup {0}  --nlogfreq {1}    --logfile {2} --verbose {3}  --input_topic {4} '.format(
                    arg.consumergroup + 'test' + arg.mode, arg.nlogfreq,
                    logfolder + '/stream_test_' + str(
                        arg.consumergroup) + '.txt', arg.verbose,
                    arg.input_topic), 'nproc': arg.nproc,
                'mem_available_total'        : 2000 * Mb,
                'cpu_usage_total'            : 95.0}
        CMDS = [pars['proc_cmd']] * pars['nproc']

    else:
        printlog('No configuration was selected')
        sys.exit(1)

    printlog(arg.name, 'parameters', pars)

    ############## RUN Monitor ################################################
    monitor()