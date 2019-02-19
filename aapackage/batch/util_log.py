# -*- coding: utf-8 -*-
"""
Implement Interface functions for logging
   writign on disk
   display time

https://docs.python-guide.org/writing/logging/
https://docs.python.org/3/howto/logging-cookbook.html


+ WRITE small test_util_log.py
check here
https://github.com/arita37/a_aapackage/blob/master/aapackage/batch/batch_daemon_launch_cli.py



>>> import logging
>>> root = logging.getLogger()
>>> root.setLevel(logging.DEBUG)
>>> handler = logging.StreamHandler()
>>> bf = logging.Formatter('{asctime} {name} {levelname:8s} {message}',
...                        style='{')
>>> handler.setFormatter(bf)
>>> root.addHandler(handler)
>>> logger = logging.getLogger('foo.bar')
>>> logger.debug('This is a DEBUG message')
2010-10-28 15:11:55,341 foo.bar DEBUG    This is a DEBUG message
>>> logger.critical('This is a CRITICAL message')
2010-10-28 15:12:11,526 foo.bar CRITICAL This is a CRITICAL message
>>> df = logging.Formatter('$asctime $name ${levelname} $message',
...                        style='$')
>>> handler.setFormatter(df)
>>> logger.debug('This is a DEBUG message')
2010-10-28 15:13:06,924 foo.bar DEBUG This is a DEBUG message
>>> logger.critical('This is a CRITICAL message')
2010-10-28 15:13:11,494 foo.bar CRITICAL This is a CRITICAL message

"""
import os
import sys
import socket
import time
import arrow
import logging

################### Logs ######################################################
global APP_ID, APP_ID2

APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
APP_ID2 = str(os.getpid()) + '_' + str(socket.gethostname())

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logfile.log")


def printlog(s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='', s9='', s10='',
             app_id='', logfile=None):
    try:
        if app_id != "":
            prefix = app_id + ',' + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
        else:
            prefix = APP_ID + ',' + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
        s = ','.join([prefix, str(s), str(s1), str(s2), str(s3), str(s4), str(s5),
                      str(s6), str(s7), str(s8), str(s9), str(s10)])

        print(s)
        writelog(s, logfile)
    except Exception as e:
        print(e)
        writelog(e, logfile)


def writelog(m="", f=None):
    f = LOG_FILE if f is None else f
    with open(f, 'a') as _log:
        _log.write(m + "\n")



def logger_setup(name=__name__, level=None):
    # logger defines
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger




""""
logging.basicConfig(level=logging.INFO)


# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)


"""
