"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os
from argparse import ArgumentParser


import numpy as np
import tensorflow as tf


####################################################################################################
####################################################################################################
from config import get_config


#### TF
from equation import get_equation as get_equation_tf
from solver import FeedForwardModel as FFtf


#### PYTORCH
from equation_tch import get_equation as get_equation_tch
from solver_tch import train




####################################################################################################
def load_argument() :
   p = ArgumentParser()
   p.add_argument("--problem_name", type=str, default='HJB')
   p.add_argument("--num_run", type=int, default=1)
   p.add_argument("--log_dir", type=str, default='./logs')
   p.add_argument("--framework", type=str, default='tf')
   arg = p.parse_args()
   return arg

# arg = tf.app.arg.arg
# tf.app.arg.DEFINE_string("problem_name", "HJB", """The name of partial differential equation.""")
# tf.app.arg.DEFINE_integer(
#    "num_run", 1, """The number of experiments to repeatedly run for the same problem."""
#)
#tf.app.arg.DEFINE_string(
#    "log_dir", "./logs", """Directory where to write event logs and output array."""
#)


def log(s):
    logging.info(s)


def main():
    arg = load_argument() 
    print(arg)
    c = get_config(arg.problem_name)


    if not os.path.exists(arg.log_dir):
        os.mkdir(arg.log_dir)

    path_prefix = os.path.join(arg.log_dir, arg.problem_name)
    with open(path_prefix + ".json", "w") as outfile:
        json.dump(
            dict( (name, getattr(c, name)) for name in dir(c) if not name.startswith("__")),
            outfile,indent=2,
        )


    if arg.framework == 'tf':
        bsde = get_equation_tf(arg.problem_name, c.dim, c.total_time, c.num_time_interval)

    elif arg.framework == 'tch':
        bsde = get_equation_tch(arg.problem_name, c.dim, c.total_time, c.num_time_interval)

    print("Running ", arg.problem_name, " on: ", arg.framework)
    print(bsde)

    #### Loop over run
    for idx_run in range(1, arg.num_run + 1):

        log("Begin to solve %s with run %d" % (arg.problem_name, idx_run))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if arg.framework == 'tf':
            tf.reset_default_graph()
            with tf.Session() as sess:
                model = FFtf(c, bsde, sess)
                model.build()
                training_history = model.train()

        elif arg.framework == 'tch':
            training_history = train(c, bsde)


        if bsde.y_init:
            log("% error of Y0: %s{:.2%}".format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init),)

            # save training history
            np.savetxt(
              "{}_training_history_{}.csv".format(path_prefix, idx_run),
              training_history,
              fmt=["%d", "%.5e", "%.5e", "%d"],
              delimiter=",",
              header="step,loss_function,target_value,elapsed_time",
              comments="",
            )




if __name__ == "__main__":
    main()
