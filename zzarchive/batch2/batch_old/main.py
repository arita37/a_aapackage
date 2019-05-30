# -*- coding: utf-8 -*-
import configparser
import os
import socket
import subprocess
import sys
from time import sleep

import boto
import pandas as pd
from boto.ec2.connection import EC2Connection

import util
from monitor_task import monitor

DIR1 = os.getcwd()
DIR_package = os.getcwd() + "/aapackage/aws/"




############################################################################################################
if __name__ == "__main__":
    # Running Background, Loop on Tasks:  ------------------------------------------------------
    MAX_BID = 0.005
    REFRESH_TIME = 60 * 5

    from . import tasklist  # Start the Celery Scheduler

    while True:
        task_pending = task_get_pending()
        monitor_add_instance_toworker()

        if len(task_pending) > 0:
            inst_active = instance_get_allactive()
            if len(inst_active) == 0:
                monitor_launch_fromtask(task_pending)
            else:
                monitor_sendto_instance(inst_active)

        task_pending = task_get_pending()
        task_active = task_get_active()

        # if len(task_pending) == 0   : instance_spot_request_cancel_all()

        if len(task_active) == 0:
            instance_spot_stop_all()
            instance_spot_request_cancel_all()

        sleep(REFRESH_TIME)
