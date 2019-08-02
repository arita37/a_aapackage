import logging
import time
import copy
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float32
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

from submodels import subnetwork_bidirectionalattn, subnetwork_dila, subnetwork_ff
from submodels import subnetwork_lstm, subnetwork_lstm_attn

tf.enable_eager_execution()


def log(s):
    logging.info(s)


"""

Variance Realized =  Sum(ri*:2)
rI**2 =   Sum(wi.ri)**2

Return = sum(ri) = Total return


"""

export_folder = "/home/ubuntu/zs3drive/"


###################################################################################################
class FeedForwardModel(object):
    """
       Global model over the full time steps
    """

    def __init__(self, config, bsde, sess, usemodel):
        self._config = config
        self._bsde = bsde  # BSDE Equation
        self._sess = sess  # TF session
        self._usemodel = usemodel

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        self._smooth = 1e-8

        self._is_training = tf.placeholder(tf.bool)
        if usemodel == 'attn':
            self.subnetwork = subnetwork_lstm_attn(config)

        if usemodel == 'lstm':
            self.subnetwork = subnetwork_lstm(config)

        if usemodel == 'ff':
            self.subnetwork = subnetwork_ff(config, self._is_training)

        if usemodel == 'dila':
            self.subnetwork = subnetwork_dila(config)

        if usemodel == 'biattn':
            self.subnetwork = subnetwork_bidirectionalattn(config)

        self._extra_train_ops = []

        # self._config.num_iterations = None  # "Nepoch"
        # self._train_ops = None  # Gradient, Vale,

    def train(self):
        t0 = time.time()
        train_history = []  # to save iteration results

        ## Validation DATA : Brownian part, drift part from MC simulation
        # dw_valid, x_valid = self._bsde.sample(self._config.batch_size)

        #################################################################
        dw_valid, x_valid = self.generate_feed()

        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._train_ops = tf.group([self._train_ops, update_ops])

        self._sess.run(tf.global_variables_initializer())  # initialization
        # begin sgd iteration
        val_writer = tf.summary.FileWriter('logs', self._sess.graph)
        merged = tf.summary.merge_all()
        for step in range(self._config.num_iterations + 1):
            # Generate MC sample AS the training input
            dw_train, x_train = self.generate_feed()

            self._sess.run(
                self._train_ops,
                feed_dict={self._dw: dw_train,
                           self._x: x_train, self._is_training: True},
            )

            ### Validation Data Eval.
            if step % self._config.logging_frequency == 0:
                loss, init, summary = self._sess.run([self._loss, self._y_init, merged],
                                                     feed_dict=feed_dict_valid)
                dt0 = time.time() - t0 + self._t_build
                train_history.append([step, loss, init, dt0])
                print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"
                      % (step, loss, init, dt0))
                val_writer.add_summary(summary, step)

        return np.array(train_history)

    def validation_train(self, sess, feed_dict_valid, train_history, merged, t0, step, val_writer):
        loss, init, summary = self._sess.run([self._loss, self._y_init, merged],
                                             feed_dict=feed_dict_valid)
        dt0 = time.time() - t0 + self._t_build
        train_history.append([step, loss, init, dt0])
        print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"
              % (step, loss, init, dt0))
        val_writer.add_summary(summary, step)

    def generate_feed(self):
        #################################################################
        dw_valid, x_valid = [], []
        for clayer in range(self._config.clayer):
            dw, x = self._bsde.sample(self._config.batch_size, clayer)
            dw_valid.append(dw)
            x_valid.append(x)

        dw_valid, x_valid = np.stack(dw_valid, axis=2), np.stack(x_valid, axis=2)
        dw_valid = np.reshape(dw_valid,
                              [self._config.batch_size,
                               self._config.clayer * self._dim, self._num_time_interval])

        x_valid = np.reshape(x_valid,
                             [self._config.batch_size,
                              self._config.clayer * self._dim, self._num_time_interval + 1])
        ##################################################################    
        return dw_valid, x_valid

    def train2(self):
        t0 = time.time()
        train_history = []  # to save iteration results

        ## Validation DATA : Brownian part, drift part from MC simulation

        #################################################################
        dw_valid, x_valid = self.generate_feed()
        ##################################################################                      
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._train_ops = tf.group([self._train_ops, update_ops])

        self._sess.run(tf.global_variables_initializer())  # initialization
        # begin sgd iteration
        val_writer = tf.summary.FileWriter('logs', self._sess.graph)
        merged = tf.summary.merge_all()
        x_all, y_all, z_all, p_all, w_all = [], [], [], [], []
        for step in range(self._config.num_iterations + 1):
            # Generate MC sample AS the training input

            dw_train, x_train = self.generate_feed()

            _ = self._sess.run(self._train_ops,
                               feed_dict={self._dw: dw_train,
                                          self._x: x_train, self._is_training: True},
                               )

            # y_all.append(y)

            ### Validation Data Eval.
            if step % self._config.logging_frequency == 0:
                loss, init, summary = self._sess.run([self._loss, self._y_init, merged],
                                                     feed_dict=feed_dict_valid)
                dt0 = time.time() - t0 + self._t_build
                train_history.append([step, loss, init, dt0])
                print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"
                      % (step, loss, init, dt0))
                val_writer.add_summary(summary, step)

        print("Writing path history on disk, {}/".format(export_folder))
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        np.save(os.path.join(export_folder, 'x.npy'), np.concatenate(x_all, axis=0))
        # np.save(export_folder + '/y.npy', np.concatenate(y_all, axis=0))
        np.save(os.path.join(export_folder, 'z.npy'), np.concatenate(z_all, axis=0))
        np.save(os.path.join(export_folder, 'p.npy'), np.concatenate(p_all, axis=0))
        np.save(os.path.join(export_folder, 'w.npy'), np.concatenate(w_all, axis=0))
        return np.array(train_history)

    def build2(self):
        """"
           y : State
           dw : Brownian,  x : drift deterministic
           z : variance
        """
        t0 = time.time()
        TT = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        ### dim X Ntime_interval for Stochastic Process
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim * self._config.clayer, self._num_time_interval], name="dW")
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim * self._config.clayer, self._num_time_interval + 1],
                                 name="X")

        ### Initialization
        ### x : state,  Cost
        self._y_init = tf.Variable(tf.random_uniform(
            [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
            dtype=TF_DTYPE))

        ## Control
        z_init = tf.Variable(
            tf.random_uniform([1, self._dim], minval=0.01, maxval=0.3, dtype=TF_DTYPE)
            # tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE)
        )

        # P
        # p_old = tf.Variable(
        #    tf.random_uniform(shape=[self._config.batch_size], minval=0.0, maxval=0.0, dtype=TF_DTYPE)
        # )

        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        z = tf.matmul(all_one_vec, z_init)

        all_p, all_z, all_w = [], [], []
        with tf.variable_scope("forward"):
            for t in range(0, self._num_time_interval - 1):
                # y = (y_old
                #        - self._bsde.delta_t * (self._bsde.f_tf(TT[t], self._x[:, :, t], y_old, z))
                #        + tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True))
                # if t == 0 :
                #    xold = self._x[:, :, t]
                ## Neural Network per Time Step, Calculate Gradient
                ## Z = [batch, dim * clayer] -> [batch, 4] for optionprice config
                if self._usemodel == 'lstm':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'ff':
                    z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim

                elif self._usemodel == 'attn':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'dila':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'biattn':
                    z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim
                all_z.append(z)

                # w = z / tf.reduce_sum(z, -1, keepdims=True)
                ####New formulae
                if t == 0:
                    w = tf.Variable(
                        tf.random.uniform([self._config.batch_size, self._config.dim * self._config.clayer], minval=0.1,
                                          maxval=0.3, dtype=TF_DTYPE))
                else:
                    w = 0.2 + z + 1 / tf.sqrt((tf.nn.moments(
                        tf.log((self._x[:, :self._config.dim, 1:t + 1]) / (self._x[:, :self._config.dim, :t])), axes=2)[
                                                   1] + self._smooth))
                w = w / tf.reduce_sum(w, -1, keepdims=True)  ###Normalize Sum to 1
                all_w.append(w)

                # y =   tf.reduce_sum( w * (self._x[:, :, t]  / self._x[:, :, t-1]  - 1), 1, keepdims=True)
                # all_y.append(y)

                ## P = [batch]
                # if t == 0:
                #    p = p_old
                # else:
                # p =  p_old * (1 + tf.reduce_sum( w * (self._x[:, :, t] / self._x[:, :, t-1] - 1), 1))
                p = tf.reduce_sum(w * (self._x[:, :self._config.dim, t] / self._x[:, :self._config.dim, t - 1] - 1), 1)

                all_p.append(p)

                # p_old = p

            # Terminal time
            # y = (y
            #        - self._bsde.delta_t * self._bsde.f_tf(TT[-1], self._x[:, :, -2], y, z)
            #        + tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True) )
            # y = self._x[:, :, -2]
            # y =   tf.reduce_sum( w * (self._x[:, :, t]  / self._x[:, :, t-1]  - 1), 1, keepdims=True)

            # all_y.append(y)
            # w = z / tf.reduce_sum(z, -1, keepdims=True)

            p = tf.stack(all_p, axis=-1)
            # self.all_y = tf.stack(all_y, axis=-1)
            self.all_z = tf.stack(all_z, axis=-1)
            self.all_w = tf.stack(all_w, axis=-1)
            self.all_p = p

            # Final Difference :
            # delta = tf.math.reduce_variance(p, axis=1) / (1 + tf.reduce_mean(p, 1))
            # m = tf.reduce_mean(p, 1)
            # delta = tf.nn.moments(p, axes=1)[1] / ( tf.square(tf.reduce_mean(p, 1)))
            # delta = tf.nn.moments(p, axes=1)[1] / ( tf.reduce_mean(p, 1) )

            # delta = -tf.reduce_mean(p, 1) + tf.nn.moments(p, axes=1)[1] 

            #######  -Sum(ri)   +Sum(ri**2)  
            delta = -0.5 * tf.reduce_sum(p[:, 1:], 1) + tf.nn.moments(p[:, 1:], axes=1)[1]

            # use linear approximation outside the clipped range
            # self._loss = tf.reduce_mean(tf.where( tf.abs(delta) < DELTA_CLIP,
            #                            tf.square(delta),
            #                            2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2,
            # ))

            self._loss = tf.reduce_mean(delta)

        tf.summary.scalar('loss', self._loss)

        # train operations
        global_step = tf.get_variable(
            "global_step", [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int32,
        )

        learning_rate = tf.train.piecewise_constant(
            global_step, self._config.lr_boundaries, self._config.lr_values
        )

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step, name="train_step"
        )
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time() - t0

    def build(self):
        """"
           y : State
           dw : Brownian,  x : drift deterministic
           z : variance
        """
        t0 = time.time()
        TT = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        ### dim X Ntime_interval for Stochastic Process
        self._dw = tf.placeholder(TF_DTYPE, [None, self._config.clayer * self._dim, self._num_time_interval], name="dW")
        self._x = tf.placeholder(TF_DTYPE, [None, self._config.clayer * self._dim, self._num_time_interval + 1],
                                 name="X")

        ### Initialization
        ## x : state,  Cost
        self._y_init = tf.Variable(
            tf.random_uniform(
                [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
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
                y = (
                        y
                        - self._bsde.delta_t * (self._bsde.f_tf(TT[t], self._x[:, :self._dim, t], y, z))
                        + tf.reduce_sum(z * self._dw[:, :self._dim, t], 1, keepdims=True)
                )

                ## Neural Network per Time Step, Calculate Gradient
                if self._usemodel == 'lstm':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'ff':
                    z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim

                elif self._usemodel == 'attn':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'dila':
                    z = self.subnetwork.build([self._x[:, :, t + 1]], t) / self._dim

                elif self._usemodel == 'biattn':
                    z = self.subnetwork.build(self._x[:, :, t + 1], t) / self._dim

            # Terminal time
            y = (
                    y
                    - self._bsde.delta_t * self._bsde.f_tf(TT[-1], self._x[:, :self._dim, -2], y, z)
                    + tf.reduce_sum(z * self._dw[:, :self._dim, -1], 1, keepdims=True)
            )

            # Final Difference :
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :self._dim, -1])

            # use linear approximation outside the clipped range
            self._loss = tf.reduce_mean(
                tf.where(
                    tf.abs(delta) < DELTA_CLIP,
                    tf.square(delta),
                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2,
                )
            )
        tf.summary.scalar('loss', self._loss)

        # train operations
        global_step = tf.get_variable(
            "global_step", [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int32,
        )

        learning_rate = tf.train.piecewise_constant(
            global_step, self._config.lr_boundaries, self._config.lr_values
        )

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step, name="train_step"
        )
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time() - t0
