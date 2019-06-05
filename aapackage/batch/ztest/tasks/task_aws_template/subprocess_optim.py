#!/bin/python
"""

This is a dummy optimizer, designed to be executed in a independend python process.

Notable interactions:

loads the file "parameters.toml", located at its same folder.

outputs "results.txt" file, containing an array that can be sourced inside python.

Those interactions are defined in batch_sequencer.py and should be conserved among all optimizers.


"""
import os
import sys

import arrow
import pandas as pd
import toml
from scipy import optimize

###########################################################################################
BATCH_RESULT = "../../ztest/batch_results"


APP_ID = __file__ + "," + str(os.getpid()) + ","


def printlog(s="", s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9="", s10=""):
    try:
        prefix = APP_ID + "," + arrow.utcnow().to("Japan").format("YYYYMMDD_HHmmss,")
        s = ",".join(
            [
                prefix,
                str(s),
                str(s1),
                str(s2),
                str(s3),
                str(s4),
                str(s5),
                str(s6),
                str(s7),
                str(s8),
                str(s9),
                str(s10),
            ]
        )

        # logging.info(s)
        print(s)
        log(s)
    except Exception as e:
        # logging.info(e)
        print(e)


def save_results(s1):
    with open(BATCH_RESULT + "/result%i.txt" % ii, "a") as ResultOutput:
        result = str(list(s1))
        ResultOutput.write(result + "\n")


###########################################################################################
###########################################################################################


def optimizerFunction(x):
    x1, x2, x3, x4 = x
    delta = x1 ** 2 * x4
    omega = -500 + delta * x1 + x2

    return omega


def execute(ii, params_dict):
    res = optimize.minimize(
        optimizerFunction,
        [params_dict["x1"], params_dict["x2"], params_dict["x3"], params_dict["x4"]],
    )
    print("Result: %s" % res.x)

    save_results(res.x)
    print("Finished Program: %s" % str(os.getpid()))


if __name__ == "__main__":
    ii = int(sys.argv[1])

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isdir(BATCH_RESULT):
        os.mkdir(BATCH_RESULT)
    hyper_parameter_set = pd.read_csv("hyperparams.csv")
    params_dict = hyper_parameter_set.iloc[ii].to_dict()

    execute(ii, params_dict)


###########################################################################################
###########################################################################################
