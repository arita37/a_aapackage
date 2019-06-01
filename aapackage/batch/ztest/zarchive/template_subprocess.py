# -*- coding: utf-8 -*-
"""
hyperparams
  file_log
  file_data
  file_data_method
  


"""
import os
import sys

import numpy as np
import pandas as pd

import pygmo as pg

if __name__ != "__main__":
    sys.exit(0)


###############################################################################
###############################################################################
hyperparam_file = "hyperparams.csv"
hh_exclude = []


###############################################################################
###############################################################################
def load_arguments():
    import argparse

    ppa = argparse.ArgumentParser()  # Command Line input
    ppa.add_argument("--ii", type=int, default=0, help="test / test02")

    arg = ppa.parse_args()
    return arg


def load_data_session(file_data, method="spyder"):
    if method == "spyder":
        from spyderlib.utils.iofuncs import load_dictionary

        globals().update(load_dictionary(filedata)[0])

    if method == "shelve":
        import shelve

        "dict of var in myshelf.db"
        with shelve.open(file_data, flag="r") as db:
            for key in db:
                print(key)
                globals()[key] = db[key]


def logs(ss):
    ss_head = []
    ss2 = ",".join([str(x) for x in ss])
    print(ss2)

    with open(logfile, "a") as f1:
        f1.write(ss2)


###############################################################################
##### Hyper-params  ###########################################################
args = load_arguments()
hh = pd.read_csv(hyperparam_file).iloc[args.ii, :]  # hyper params
logfile = hh["file_log"]

for x in hh.columns:
    if x not in hh_exclude:
        print(x, hh[x])
        # globals()[x] = x


##### Session data   ##########################################################
load_data_session(hh["file_data"], method=hh["file_data_method"])


###############################################################################
##### Scripts  ################################################################
logs("Start Script", __file__)


# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
prob = pg.problem(pg.schwefel(10))

# 2 - Instantiate a pagmo algorithm
algo = pg.algorithm(pg.sade(gen=10))

# 3 - Instantiate an archipelago with 16 islands having each 20 individuals
archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=10)

# 4 - Run the evolution in parallel on the 16 separate islands 10 times.
archi.evolve(5)

# 5 - Wait for the evolutions to be finished
archi.wait()

# 6 - Print the fitness of the best solution in each island
res = [isl.get_population().champion_f for isl in archi]
print(res)


###############################################################################
##### End Script Rnme Folder ##################################################
#### Rename folder ################
folder_par = ""
folder_par2 = ""
# os.rename( folder_par, folder_par2  )
