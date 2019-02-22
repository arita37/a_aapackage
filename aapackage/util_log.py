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


logging.basicConfig(level=logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)


"""
import os
import sys
import socket
import time
import arrow
import logging
from logging.handlers import TimedRotatingFileHandler


################### Logs #################################################################
APP_ID  = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
APP_ID2 = str(os.getpid()) + '_' + str(socket.gethostname())

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logfile.log")
FORMATTER_1  = logging.Formatter( "%(asctime)s — %(name)s — %(levelname)s — %(message)s" )
FORMATTER_2 = logging.Formatter( '%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s'   )
#LOG_FILE = "my_app.log"


###############################################################################
def create_appid(filename ) :
   appid  = filename + ',' + str(os.getpid()) + ',' + str( socket.gethostname() )
   return appid


def create_uniqueid() :
   arrow.utcnow().to('Japan').format("_YYYYMMDDHHmmss_")  + str( random.randint(1000, 9999))


##########################################################################################
################### Print ################################################################
def printlog( s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='', s9='', s10='',
              app_id='', logfile=None, iswritelog=True):
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


########################################################################################
################### Logger #############################################################
def logger_setup(logger_name, log_file=None, formatter=None, isrotate=False):
   """
    my_logger = util_log.logger_setup("my module name", log_file="")
    APP_ID    = util_log.create_appid(__file__ )
    def log(s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='', s9='', s10='') :
       my_logger.debug( ",".join( [APP_ID, str(s1), str(s2), str(s3), str(s4), str(s5) ,
                        str(s6), str(s7), str(s8), str(s9), str(s10)] ) )
   """
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG)      # better to have too much log than not enough
   logger.addHandler( logger_handler_console( formatter ) )
   logger.addHandler( logger_handler_file( formatter=formatter, log_file_used=log_file) )
   # with this pattern, it's rarely necessary to propagate the error up to parent
   logger.propagate = False
   return logger


def logger_setup2(name=__name__, level=None):
    # logger defines
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def logger_handler_console(formatter=None ):
   formatter = FORMATTER_1 if formatter is None else formatter
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter( formatter )
   return console_handler


def logger_handler_file(isrotate=False, rotate_time='midnight', formatter=None, log_file_used=None):
   formatter = FORMATTER_1 if formatter is None else formatter
   log_file_used = LOG_FILE if log_file_used is None else log_file_used
   if isrotate :
     fh = TimedRotatingFileHandler( log_file_used, when=rotate_time)
     fh.setFormatter( formatter )
     return fh
   else :
     fh = logging.FileHandler(log_file_used)
     fh.setFormatter( formatter )
     return fh





