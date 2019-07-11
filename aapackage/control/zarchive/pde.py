"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os
from argparse import ArgumentParser


import numpy as np
####################################################################################################



####################################################################################################
def load_argument() :
   p = ArgumentParser()
   p.add_argument("--problem_name", type=str, default='HJB')
   p.add_argument("--num_run", type=int, default=1)
   p.add_argument("--log_dir", type=str, default='./logs')
   p.add_argument("--framework", type=str, default='tf')
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
            dict( (name, getattr(conf, name)) for name in dir(conf) if not name.startswith("__")),
            outfile, indent=2,
        )    
    

def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")



import numpy as np


class Config(object):
    n_layer = 4
    batch_size = 64
    valid_size = 256
    step_boundaries = [2000, 4000]
    num_iterations = 6000
    logging_frequency = 10
    verbose = True
    y_init_range = [0, 1]


class HJBConfig(Config):
    # Y_0 is about 4.5901.
    dim = 2
    total_time = 1.0
    num_time_interval = 10
    lr_boundaries = [400]
    num_iterations = 20
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]






import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class HJB(Equation):
    """
       nsample x VecDim x dT
       
      
    
    """
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lambda = 1.0

    def sample(self, num_sample):

        ### All the samples
        dw_sample = (
            normal.rvs(size=[num_sample, self._dim, self._num_time_interval]) * self._sqrt_delta_t
        )

        ### Euler Discretization
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init

        ### Euler Sequential ; Diffusion process
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        # Constraints 1
        return -self._lambda * tf.reduce_sum(tf.square(z), 1, keepdims=True)

    def g_tf(self, t, x):
        # Constraints 2
        return tf.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)





import logging
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0



class FeedForwardModel(object):
    """The fully connected neural network model."""

    def __init__(self, config, bsde, sess):
        self._config = config
        self._bsde = bsde  # BSDE Equation
        self._sess = sess  # TF session

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        # ops for statistics update of batch normalization
        self._extra_train_ops = []

        # self._config.num_iterations = None  # "Nepoch"
        # self._train_ops = None  # Gradient, Vale,

    def train(self):
        t0 = time.time()
        training_history = []  # to save iteration results

        ## Validation DATA : Brownian part, drift part from MC simulation
        dw_valid, x_valid = self._bsde.sample(self._config.valid_size)
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        self._sess.run(tf.global_variables_initializer())  # initialization
        # begin sgd iteration
        for step in range(self._config.num_iterations + 1):
            # Generate MC sample AS the training input
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            self._sess.run(
                self._train_ops,
                feed_dict={self._dw: dw_train, 
                           self._x : x_train, 
                           self._is_training: True},
            )
            
            
            ### Validation Data Eval.
            if step % self._config.logging_frequency == 0:
                dt= time.time() - t0 + self._t_build
                loss, init = self._sess.run([self._loss, self._y_init], feed_dict=feed_dict_valid)
                training_history.append([step, loss, init, dt])
                log("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"% (step, loss, init, dt)  )


        return np.array(training_history)

    def build(self):
        """"
           y : State
           dw : Brownian,  x : drift deterministic
           z : variance

        """
        t0 = time.time()
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        ### dim X Ntime_interval for Stochastic Process
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name="dW")
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name="X")
        self._is_training = tf.placeholder(tf.bool)

        ### Initialization
        ## x : state
        ### Cost
        self._y_init = tf.Variable(
            tf.random_uniform(
                [1],
                minval=self._config.y_init_range[0],
                maxval=self._config.y_init_range[1],
                dtype=TF_DTYPE,
            )
        )
        ## Control
        z_init = tf.Variable(
            tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE)
        )
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        y = all_one_vec * self._y_init
        z = tf.matmul(all_one_vec, z_init)

        with tf.variable_scope("forward"):
            for t in range(0, self._num_time_interval - 1):
                y = (   y
                        - self._bsde.delta_t * (self._bsde.f_tf(time_stamp[t], self._x[:, :, t], y, z))
                        + tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True)
                )

                ## Neural Network per Time Step, Calculate Gradient
                z = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim

            # Terminal time
            y = (   y
                    - self._bsde.delta_t * self._bsde.f_tf(time_stamp[-1], self._x[:, :, -2], y, z)
                    + tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True)
            )

            # Final Difference :
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])

            # use linear approximation outside the clipped range
            self._loss = tf.reduce_mean(  tf.where( tf.abs(delta) < DELTA_CLIP,
                                          tf.square(delta),
                                          2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2,
                )
            )

        # train operations
        global_step = tf.get_variable(  "global_step",  [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int32,
        )

        learning_rate = tf.train.piecewise_constant(
            global_step, self._config.lr_boundaries, self._config.lr_values
        )

        ##### Loss is specific 
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), 
            global_step=global_step, name="train_step"
        )
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time() - t0


    def _subnetwork(self, x, name):
        """
          FeedForward network, connected by Input.

          FInal layer no Activation function,
          dim : nb of assets.

          num_hiddens = [dim, dim+10, dim+10, dim]
        """
        with tf.variable_scope(name):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = self._batch_norm(x, name="path_input_norm")
            for i in range(1, len(self._config.num_hiddens) - 1):
                hiddens = self._dense_batch_layer(
                    hiddens,
                    self._config.num_hiddens[i],
                    activation_fn=tf.nn.relu,
                    name="layer_{}".format(i),
                )
            output = self._dense_batch_layer(
                hiddens, self._config.num_hiddens[-1], activation_fn=None, name="final_layer"
            )
        return output

    def _dense_batch_layer(
            self, input_, output_size, activation_fn=None, stddev=5.0, name="linear"
    ):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable(
                "Matrix",
                [shape[1], output_size],
                TF_DTYPE,
                tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size)),
            )
            hiddens = tf.matmul(input_, weight)
            hiddens_bn = self._batch_norm(hiddens)

        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn

    def _batch_norm(self, x, affine=True, name="batch_norm"):
        """Batch normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                "beta",
                params_shape,
                TF_DTYPE,
                initializer=tf.random_normal_initializer(0.0, stddev=0.1, dtype=TF_DTYPE),
            )

            gamma = tf.get_variable(
                "gamma",
                params_shape,
                TF_DTYPE,
                initializer=tf.random_uniform_initializer(0.1, 0.5, dtype=TF_DTYPE),
            )

            moving_mean = tf.get_variable(
                "moving_mean",
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(0.0, TF_DTYPE),
                trainable=False,
            )

            moving_variance = tf.get_variable(
                "moving_variance",
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(1.0, TF_DTYPE),
                trainable=False,
            )

            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name="moments")
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, MOMENTUM)
            )
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, MOMENTUM)
            )
            mean, variance = tf.cond(
                self._is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance)
            )
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y




def main():
    arg = load_argument() 
    print(arg)
    c = get_config(arg.problem_name)


    log_init(arg.log_dir)
    path_prefix = os.path.join(arg.log_dir, arg.problem_name)
    config_dump(c, path_prefix)     


    if arg.framework == 'tf':
        import tensorflow as tf
        from equation import get_equation as get_equation_tf 
        from solver import FeedForwardModel as FFtf

        bsde = get_equation_tf(arg.problem_name, c.dim, c.total_time, c.num_time_interval)


    print("Running ", arg.problem_name, " on: ", arg.framework)
    print(bsde)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-6s %(message)s")
    
    #### Loop over run
    for k in range(1, arg.num_run + 1):
        log("Begin to solve %s with run %d" % (arg.problem_name, k))
        log("Y0_true: %.4e" % bsde.y_init) if bsde.y_init else None
        if arg.framework == 'tf':
            tf.reset_default_graph()
            with tf.Session() as sess:
                model = FFtf(c, bsde, sess)
                model.build()
                training_history = model.train()

        if bsde.y_init:
            log("% error of Y0: %s{:.2%}".format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init),)

        # save training history
        np.savetxt(
          "{}_training_history_{}.csv".format(path_prefix, k),
          training_history,
          fmt=["%d", "%.5e", "%.5e", "%d"],
          delimiter=",",
          header="step,loss_function,target_value,dt",
          comments="",
        )




if __name__ == "__main__":
    main()
















