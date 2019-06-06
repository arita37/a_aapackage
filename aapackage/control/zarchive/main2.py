"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).
"""

import json
import logging
import os

import numpy as np


####################################################################################################
####################################################################################################
from config import get_config




from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--problem_name", type=str, default='AllenCahn')
parser.add_argument("--num_run", type=int, default=1)
parser.add_argument("--log_dir", type=str, default='./logs')
parser.add_argument("--framework", type=str, default='tch')


def log(s):
    logging.info(s)


def main():
    args = parser.parse_args()
    config = get_config(args.problem_name)

    if args.framework == 'tf':
        import tensorflow as tf
        from equation import get_equation as get_equation_tf
        from solver import FeedForwardModel as FFtf
        bsde = get_equation_tf(args.problem_name, config.dim, config.total_time, config.num_time_interval)


    elif args.framework == 'tch':
        from equation_tch import get_equation as get_equation_tch
        from solver_tch import train
        
        bsde = get_equation_tch(args.problem_name, config.dim, config.total_time, config.num_time_interval)



    print("Running ", args.problem_name, " on: ", args.framework)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    path_prefix = os.path.join(args.log_dir, args.problem_name)
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
    for idx_run in range(1, args.num_run + 1):

        log("Begin to solve %s with run %d" % (args.problem_name, idx_run))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if args.framework == 'tf':
            tf.reset_default_graph()
            with tf.Session() as sess:
                model = FFtf(config, bsde, sess)
                model.build()
                training_history = model.train()

        elif args.framework == 'tch':
            training_history = train(config, bsde)

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