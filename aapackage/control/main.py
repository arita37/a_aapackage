"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os

import numpy as np
import tensorflow as tf

####################################################################################################
####################################################################################################
from config import get_config
from equation import get_equation as get_equation_tf

from equation_tch import get_equation as get_equation_tch

from solver import FeedForwardModel as FFtf
from solver_tch import train

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--problem_name", type=str, default='AllenCahn')
parser.add_argument("--num_run", type=int, default=1)
parser.add_argument("--log_dir", type=str, default='./logs')
parser.add_argument("--framework", type=str, default='tf')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("problem_name", "HJB", """The name of partial differential equation.""")
tf.app.flags.DEFINE_integer(
    "num_run", 1, """The number of experiments to repeatedly run for the same problem."""
)
tf.app.flags.DEFINE_string(
    "log_dir", "./logs", """Directory where to write event logs and output array."""
)


def log(s):
    logging.info(s)


def main():
    FLAGS = parser.parse_args()
    config = get_config(FLAGS.problem_name)

    if FLAGS.framework == 'tf':
        bsde = get_equation_tf(FLAGS.problem_name, config.dim, config.total_time, config.num_time_interval)
    elif FLAGS.framework == 'tch':
        bsde = get_equation_tch(FLAGS.problem_name, config.dim, config.total_time, config.num_time_interval)

    print("Running ", FLAGS.problem_name, " on: ", FLAGS.framework)

    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.problem_name)
    with open("{}_config.json".format(path_prefix), "w") as outfile:
        json.dump(
            dict(
                (name, getattr(config, name)) for name in dir(config) if not name.startswith("__")
            ),
            outfile,
            indent=2,
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")

    #### Loop over run
    for idx_run in range(1, FLAGS.num_run + 1):
        tf.reset_default_graph()
        log("Begin to solve %s with run %d" % (FLAGS.problem_name, idx_run))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if FLAGS.framework == 'tf':
            with tf.Session() as sess:
                model = FFtf(config, bsde, sess)
                model.build()
                training_history = model.train()

        elif FLAGS.framework == 'tch':
            train(config, bsde)

            if bsde.y_init:
                log(
                    "relative error of Y0: %s",
                    "{:.2%}".format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init),
                )

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
