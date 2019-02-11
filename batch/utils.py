# -*- coding: utf-8 -*-

import psutil
from time import sleep
import util_log


def ps_wait_process_complete(subprocess_list):
    for pid in subprocess_list:
        while True:
            try:
                pr = psutil.Process(pid)
                try:
                    pr_status = pr.status()
                except TypeError:  # psutil < 2.0
                    pr_status = pr.status
                except psutil.NoSuchProcess:  # pragma: no cover
                    break
                # Check if process status indicates we should exit
                if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    break
            except:
                break
            sleep(1)


def logs(*args, **kwargs):
    log_string = "{0}, {1}"
    log_string = log_string.format(", ".join(kwargs.values()), ", ".join(args))
    print(log_string)
    util_log.log(m=log_string)

def os_getparent():
    pass
