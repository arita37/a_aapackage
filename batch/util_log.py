# -*- coding: utf-8 -*-
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

LOG_FILE = "logfile.log"

logging.basicConfig(level=logging.INFO)


def printlog(s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='', s9='', s10=''):
    try:
        prefix = APP_ID + ',' + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
        s = ','.join([prefix, str(s), str(s1), str(s2), str(s3), str(s4), str(s5),
                      str(s6), str(s7), str(s8), str(s9), str(s10)])

        # logging.info(s)
        print(s)
        return s
    except Exception as e:
        # logging.info(e)
        print(e)


def log(m="", f=None):
    f = LOG_FILE if f is None else f
    with open(f, 'a') as _log:
        _log.write(m)


""""

# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)


"""