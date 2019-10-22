"""
source activate py36c
cd control



For fully connected layers:
python  main.py  --train --problem_name PricingOption  --usemodel ff

For global lstm layer:
python  main.py --do train --problem_name PricingOption  --usemodel lstm

For global lstm with attention:
python  main.py  --do train --problem_name PricingOption  --usemodel attn



### Not yet tested
For global dilated rnn:
python main.py --do train --problem_name PricingOption --usemodel dila

For global bi-directional rnn with attention
python main.py --do train --problem_name PricingOption --usemodel biattn


To predict sequences from disk, use:
python main.py --do predict --usemodel lstm --array_dir='path/to/directory/where/x.npy/and/dw.npy/are/located'


For tenosrboard, run this from the 'control' directory.
tensorboard   --logdir=logs/

To use old build() and train() methods, set clayer to 1 in config file.



"""
import json
import logging
import os, sys
from argparse import ArgumentParser
import numpy as np

####################################################################################################
from config import get_config, export_folder



# De activate GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



####################################################################################################
def load_argument():
    p = ArgumentParser()
    p.add_argument("--do", type=str, default='train/predict/generate paths')

    p.add_argument("--problem_name", type=str, default='PricingOption')
    p.add_argument("--num_run", type=int, default=1)
    p.add_argument("--log_dir", type=str, default='./logs')
    p.add_argument("--framework", type=str, default='tf')
    p.add_argument("--usemodel", type=str, default='lstm')

    p.add_argument("--input_folder", type=str, default='in_default/')
    p.add_argument("--output_folder", type=str, default='out_default/')

    arg = p.parse_args()
    return arg


def log(s):
    logging.info(s)


def log_init(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


def config_dump(conf, path_prefix):
    with open(path_prefix + ".json", "w") as outfile:
        json.dump(
            dict((name, getattr(conf, name)) for name in dir(conf) if not name.startswith("__")),
            outfile, indent=2,
        )


def tf_save(tf, sess, folder, modelname="model.ckpt"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    saver = tf.train.Saver()
    save_path = saver.save(sess,   folder + "/" + modelname )
    print("TensorFlow Checkpoints saved in {}".format(save_path))


def main():
    arg = load_argument()
    print(arg)
    c = get_config(arg.problem_name)

    log_init(arg.log_dir)
    path_prefix = os.path.join(arg.log_dir, arg.problem_name)
    config_dump(c, path_prefix)

    # Get generator
    if arg.framework == 'tf':
        import tensorflow as tf
        from equation import get_equation as get_equation_tf
        from solver import globalModel as FFtf

        bsde = get_equation_tf(arg.problem_name, c.dim, c.total_time, c.num_time_interval)


    elif arg.framework == 'tch':
        from equation_tch import get_equation as get_equation_tch
        from solver_tch import train
        bsde = get_equation_tch(arg.problem_name, c.dim, c.total_time, c.num_time_interval)

    print("Running ", arg.problem_name, " on: ", arg.framework)
    print(bsde)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")


    #### Generated Sample path  ONLY
    if arg.do == "generate_path" :
        bsde.sample_save()
        sys.exit(0)

    #### Loop over each epoch
    for k in range(1, arg.num_run + 1):
        log("Begin to solve %s with run %d" % (arg.problem_name, k))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if arg.framework == 'tf':

            ### Model building
            tf.reset_default_graph()
            with tf.Session() as sess:
                model = FFtf(c, bsde, sess, arg.usemodel)
                model.build2()  # model.build()


                # Training
                if arg.do == "train":
                    training_history = model.train2()  # model.train()
                    tf_save(tf, sess, export_folder + "/model/", "model.ckpt")

                elif arg.do == "predict" :
                    model.predict_sequence(arg.input_folder,
                                           arg.output_folder)


        elif arg.framework == 'tch':
            training_history = train(c, bsde, arg.usemodel)

        #if bsde.y_init:
        #    log("% error of Y0: %s{:.2%}".format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init), )

        # save training history
        if arg.do == "train":
            np.savetxt(
                "{}_training_history_{}.csv".format(path_prefix, k),
                training_history,
                fmt=["%d", "%.5e", "%.5e", "%d"],
                delimiter=",",
                header="step,loss_function,target_value,elapsed_time",
                comments="",
            )


if __name__ == "__main__":
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    main()





# arg = tf.app.arg.arg
# tf.app.arg.DEFINE_string("problem_name", "HJB", """The name of partial differential equation.""")
# tf.app.arg.DEFINE_integer(
#    "num_run", 1, """The number of experiments to repeatedly run for the same problem."""
# )
# tf.app.arg.DEFINE_string(
#    "log_dir", "./logs", """Directory where to write event logs and output array."""
# )




