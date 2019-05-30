# -*- coding: utf-8 -*-
import configparser
import os
import sys
from time import sleep

import util
from celery import Celery

from . import tasklist  # File containing all the tasks_folder

DIR1 = os.getcwd()  # DIRCWD= 'D:/_devs/Python01/project27/'
DIR_package = os.getcwd() + "/aapackage/aws/"
os.chdir(DIR1)
sys.path.append(DIR1 + "/aapackage")


######################  Initialization    #######################################################
config = configparser.ConfigParser()
config.read(DIR_package + "config.cfg")

DBNAME = DIR_package + "task_scheduler.db"
BROKER = config.get("CELERY", "BROKER")  #'amqp://localhost//'

WORKER_DIR = "/home/"  # Main Folder in Worker
S3_DIR = "/zdisk/tasks/"  # Main Folder in Worker


# -------- Module containing List of task: task01.py in folder Package ----------------------------
app = Celery("tasks_folder", broker=BROKER)
app.conf.update(
    BROKER_HEARTBEAT=10,
    CELERY_ACKS_LATE=True,
    # CELERY_RESULT_BACKEND=
    # CELERY_RESULT_DBURI=
    CELERYD_PREFETCH_MULTIPLIER=1,
    CELERY_TRACK_STARTED=True,
)

# ------  List of Pre-Defined Tasks ---------------------------------------------------------------
@app_task
def t_runscript_file(tid, script_folder, script_file, pars):

    # Create Folder Locally for Task01

    # Download folder from S3

    # Execute Script

    # Save State to S3

    return state_path


@app_task
def t_runscript_string(tid, script_folder, script_code, pars):
    return tid


@app_task
def t_runscript_parallel(tid, script_folder, script_file, pars):
    return tid


"""
app = Celery('tasks_folder', backend='amqp', broker='amqp://guest@my.server.com//')

CELERY_ACCEPT_CONTENT: ['msgpack']
BROKER_HEARTBEAT: 300
CELERY_TRACK_STARTED: True
CELERY_TIMEZONE: 'US/Pacific-New'
CELERY_SEND_TASK_SENT_EVENT: True
CELERY_RESULT_BACKEND = "amqp"
CELERY_RESULT_BACKEND: 'redis://127.0.0.1/'


BROKER_URL: 'amqp://guest:********@172.31.0.173:5672//'
BROKER_HOST = "127.0.0.1" #IP address of the server running RabbitMQ and Celery
BROKER_PORT = 5672


CELERY_ENABLE_UTC: True
CELERY_TASK_SERIALIZER: 'msgpack'
USER_RULES_DATASET_INDEX: 'user_rules'
CELERY_SEND_TASK_ERROR_EMAILS: False
CELERY_ACKS_LATE: True
CELERY_QUEUE_MAX_PRIORITY: 10
CELERY_SEND_EVENTS: True
CELERY_TASK_RESULT_EXPIRES: 86400
CELERY_IMPORTS: ['proj.job_worker']
CELERY_EVENT_SERIALIZER: 'msgpack'
CELERYD_PREFETCH_MULTIPLIER: 1
CELERY_RESULT_SERIALIZER: 'msgpack'
BROKER_HEARTBEAT_CHECKRATE: 5
CELERY_IGNORE_RESULT = False

CELERY_IMPORTS=("tasks",)

app = Celery('tasks', backend='amqp',
broker='amqp://<user>:<password>@<ip>/<vhost>')

http://avilpage.com/2014/11/scaling-celery-sending-tasks-to-remote.html

Configure RabbitMQ so that Machine B can connect to it.
# add new user
sudo rabbitmqctl add_user <user> <password>

# add new virtual host
sudo rabbitmqctl add_vhost <vhost_name>

# set permissions for user on vhost
sudo rabbitmqctl set_permissions -p <vhost_name> <user> ".*" ".*" ".*"

# restart rabbit
sudo rabbitmqctl restart
Create a new file remote.py with a simple task. Here we have broker installed in machine A. So give ip address of machine 1 in broker url option.
from celery import Celery

app = Celery('tasks', backend='amqp',
broker='amqp://<user>:<password>@<ip>/<vhost>')

def add(x, y):
return x + y

"""


####################### SEND TASK #############################################################
# 4) Module Launcher of tasks_folder -------------------------------------------------

task_list = []
for i in range(0, 2):
    task_list.append(tasklist.t_run_filescript.delay(5))
    # task_module.task_function.delay(arg1, arg2, arg3)

# All the tasks_folder will be sent right away To Rabbit Broker  and back result in task_list
len(task_list)  # Nb of tasks_folder being done

t = task_list[0]
t.ready()


#################### REMOTE MACHINE ########################################################
# 3) Launch remote worker with Name_of_Module ----------------------------------------
# File task01 must be in the worker folder
# celery worker  -A tasklist - -loglevel=info       #Launch Task file in remotev

"""
On Machine B:
Install Celery.
Copy remote.py file from machine A to this machine.
Run a worker to consume the tasks
celery worker -l info -A remote
As soon as you launch the worker, you will receive the tasks you queued up and gets executed immediately.

task01 must be accessed on remote engine and on local server, as well as the code source.

"""


"""
software -> celery:3.1.25 (Cipater) kombu:3.0.37 py:2.7.5
            billiard:3.3.0.23 py-amqp:1.4.9
platform -> system:Linux arch:64bit, ELF imp:CPython
loader   -> celery.loaders.app.AppLoader
settings -> transport:amqp results:redis://127.0.0.1/

CELERY_ACCEPT_CONTENT: ['msgpack']
BROKER_HEARTBEAT: 300
CELERY_TRACK_STARTED: True
CELERY_TIMEZONE: 'US/Pacific-New'
CELERY_SEND_TASK_SENT_EVENT: True
BROKER_URL: 'amqp://guest:********@172.31.0.173:5672//'
CELERY_ENABLE_UTC: True
CELERY_RESULT_BACKEND: 'redis://127.0.0.1/'
CELERY_TASK_SERIALIZER: 'msgpack'
USER_RULES_DATASET_INDEX: 'user_rules'
CELERY_SEND_TASK_ERROR_EMAILS: False
CELERY_ACKS_LATE: True
CELERY_QUEUE_MAX_PRIORITY: 10
CELERY_SEND_EVENTS: True
CELERY_TASK_RESULT_EXPIRES: 86400
CELERY_IMPORTS: ['proj.job_worker']
CELERY_EVENT_SERIALIZER: 'msgpack'
CELERYD_PREFETCH_MULTIPLIER: 1
CELERY_RESULT_SERIALIZER: 'msgpack'
BROKER_HEARTBEAT_CHECKRATE: 5
"""
