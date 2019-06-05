# -*- coding: utf-8 -*-
"""
Run task

python main.py  --hyperparam      --subprocess_script sub





"""
import errno
import logging
import os
import random
import socket
import sys
import time

import arrow
import numpy as np
import pandas as pd
import psutil
import toml

from aapackage import util_log

###########################################################################################
OUTFOLDER = (
    os.getcwd()
    + "/batch/ztest/out/"
    + os.path.abspath(__file__).split("/")[-2].replace("_qstart", "")
)


###########################################################################################
APP_ID = util_log.create_appid(__file__)


def log(s="", s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9=""):
    util_log.printlog(
        s="",
        s1="",
        s2="",
        s3="",
        s4="",
        s5="",
        s6="",
        s7="",
        s8="",
        s9="",
        APP_ID=APP_ID,
        LOG_FILE="log_file.log",
    )


###########################################################################################
def os_getparent(dir0):
    return os.path.abspath(os.path.join(dir0, os.pardir))


def os_chdir(filename):
    return os.chdir(os.path.dirname(os.path.realpath(filename)))


def os_python_path():
    return str(sys.executable)


def os_folder_rename(old_folder, new_folder):
    try:
        os.rename(old_folder, new_folder)
    except Exception as e:
        os.rename(old_folder, new_folder + str(random.randint(100, 999)))


def os_folder_create(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def load_data_session(file_data, method="spyder"):
    if method == "spyder":
        try:
            from spyderlib.utils.iofuncs import load_dictionary

            globals().update(load_dictionary(filedata)[0])
        except:
            pass

    if method == "shelve":
        import shelve

        "dict of var in myshelf.db"
        try:
            with shelve.open(file_data, flag="r") as db:
                for key in db:
                    print(key)
                    globals()[key] = db[key]
        except:
            pass


def save_results(BATCH_RESULT, ddict, ii, file_data):
    if not os.path.isdir(BATCH_RESULT):
        os.makedirs(BATCH_RESULT)

    try:
        import shelve

        "dict of var in myshelf.db"
        with shelve.open(file_data, flag="w") as db:
            db["res"] = ddict
    except:
        pass
    with open(BATCH_RESULT + "/result%i.txt" % ii, "a") as ff:
        result = str(ddict)
        ff.write(result + "\n")


###########################################################################################


def batch_result_folder(prefix):
    BATCH_RESULT = prefix + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "/"
    if not os.path.isdir(BATCH_RESULT):
        os.makedirs(BATCH_RESULT)

    return BATCH_RESULT
