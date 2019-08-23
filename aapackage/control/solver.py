import logging
import os
import time

import numpy as np
import tensorflow as tf

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

####################################################################################################
import sys

from config import export_folder

if not os.path.exists(export_folder):
    os.makedirs(export_folder)


def save_history(export_folder, train_history, x_all, z_all, p_all, w_all, y_all):
    print("Writing path history on disk, {}/".format(export_folder))

    w = np.concatenate(w_all, axis=0)
    x = np.concatenate(x_all, axis=0)
    p = np.concatenate(p_all, axis=0)

    np.save(os.path.join(export_folder, 'x.npy'), x)
    # np.save(export_folder + '/y.npy', np.concatenate(y_all, axis=0))
    np.save(os.path.join(export_folder, 'z.npy'), np.concatenate(z_all, axis=0))
    np.save(os.path.join(export_folder, 'p.npy'), np.concatenate(p_all, axis=0))
    np.save(export_folder + '/w.npy', w)

    save_stats(w, x, p)


def save_stats(w, x, p):
    import pandas as pd
    import matplotlib.pyplot as plt

    #### Weight at ome sample
    def get_sample(i):
        dd = {"x1": x[i][0][0][:x.shape[3] - 1],
              "x2": x[i][1][0][:x.shape[3] - 1],
              "x3": x[i][1][0][:x.shape[3] - 1],
              "pret": p[i],
              "w1": w[i][0],
              "w2": w[i][1],
              "w3": w[i][2],
              }
        df = pd.DataFrame(dd)
        return df

    dfw1 = get_sample(100000)
    dfw1.to_csv(export_folder + "/weight_sample_190k.txt")

    dfw1[["w1", "w2", "w3"]]
    plt.savefig(export_folder + "/w_sample_100k.png")
    plt.close()

    dfw1 = get_sample(10000)
    dfw1.to_csv(export_folder + "/weight_sample_10k.txt")

    dfw1[["w1", "w2", "w3"]]
    plt.savefig(export_folder + "/w_sample_10k.png")
    plt.close()

    #### Weight Convergence
    dfw = pd.DataFrame(
        {"w" + str(i + 1): w[:, i, -1] for i in range(w.shape[1])}
    )
    dfw.to_csv(export_folder + "/weight_conv.txt")

    dfw.iloc[:, :].plot()
    plt.savefig(export_folder + 'w_conv_all.png')
    plt.close()

    dfw.iloc[:10 ** 5, :].plot()
    plt.savefig(export_folder + 'w_conv_100k.png')
    plt.close()

    # get_sample( 190000 )[ [  "w1", "w2", "w3" ]   ].plot()

    #### Actual Simulation stats : correl, vol
    ### Sum(return over [0,T])
    dfx = pd.DataFrame({"x" + str(i + 1): np.sum(x[:, i, 0, :], axis=-1)
                        for i in range(x.shape[1])})

    dd = {}
    dd["ww"] = str(list(dfw.iloc[-1, :].values))
    dd["x_vol"] = {k: dfx[k].std() for k in dfx.columns}
    dd["x_corr"] = {}
    from itertools import combinations
    for x1, x2 in list(combinations(dfx.columns, 2)):
        dd["x_corr"][x1 + "_" + x2] = np.corrcoef(dfx[x1].values, dfx[x2].values)[1, 0],

    import json
    json.dump(dd, open(export_folder + "/x_stats2.txt", mode="w"))


"""

dir1 = "/home/ubuntu/aagit/aapackage/aapackage/control/numpy_arrays/"

from config import  export_folder
dir1 =  export_folder
print( export_folder)

##################################################################################
x = np.load( dir1 + "x.npy"  )
z = np.load( dir1 +"z.npy"  )
p = np.load( dir1 +"p.npy"  )
w = np.load( dir1 +"w.npy"  )

x.shape, z.shape, p.shape, w.shape

##################################################################################
def get_sample(i) :
  dd = { "x1" : x[i][0][0][:x.shape[3]-1],
   "x2" : x[i][1][0][:x.shape[3]-1],
   "x3" : x[i][1][0][:x.shape[3]-1],
   "pret" : p[i],
   "w1" : w[i][0],
   "w2" : w[i][1],
   "w3" : w[i][2],   
   "z1" : z[i][0],
   "z2" : z[i][1],   
  }  
  df = pd.DataFrame(dd)
  return df


#### Time sample
get_sample( 190000 )[ [  "w1", "w2", "w3" ]   ].plot()  



"""
dw_path = 'logs/w.npy'
x_path = 'logs/x.npy'


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
        self._T = bsde.num_time_interval
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

        ## Validation DATA : Brownian part, drift part from MC simulation
        # dw_valid, x_valid = self._bsde.sample(self._config.batch_size)
        #################################################################
        dw_valid, x_valid = self.generate_feed()
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._train_ops = tf.group([self._train_ops, update_ops])

        # initialization
        self._sess.run(tf.global_variables_initializer())
        val_writer = tf.summary.FileWriter('logs', self._sess.graph)
        merged = tf.summary.merge_all()
        train_history = []  # to save iteration results

        for step in range(self._config.num_iterations + 1):
            # Generate MC sample AS the training input
            dw_train, x_train = self.generate_feed()

            self._sess.run(self._train_ops,
                           feed_dict={self._dw: dw_train,
                                      self._x: x_train, self._is_training: True}, )

            ### Validation Data Eval.
            self.validation_train(feed_dict_valid, train_history, merged, t0, step, val_writer)

        return np.array(train_history)

    def validation_train(self, feed_dict_valid, train_history, merged, t0, step, val_writer):
        if step % self._config.logging_frequency != 0:
            return 0
        loss, init, summary = self._sess.run([self._loss, self._y_init, merged],
                                             feed_dict=feed_dict_valid)
        dt0 = time.time() - t0 + self._t_build
        train_history.append([step, loss, init, dt0])
        print("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u"
              % (step, loss, init, dt0))
        val_writer.add_summary(summary, step)

    def generate_feed(self):
        dw_valid, x_valid = [], []
        for clayer in range(self._config.clayer):
            dw, x = self._bsde.sample(self._config.batch_size, clayer)
            dw_valid.append(dw)
            x_valid.append(x)

        dw_valid, x_valid = np.stack(dw_valid, axis=2), np.stack(x_valid, axis=2)
        dw_valid = np.reshape(dw_valid,
                              [self._config.batch_size,
                               self._config.clayer * self._dim, self._T])

        x_valid = np.reshape(x_valid,
                             [self._config.batch_size,
                              self._config.clayer * self._dim, self._T + 1])
        ##################################################################
        return dw_valid, x_valid

    def train2(self):
        t0 = time.time()

        ## Validation DATA : Brownian part, drift part from MC simulation
        dw_valid, x_valid = self.generate_feed()
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)  # V1 compatibility
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._train_ops = tf.group([self._train_ops, update_ops])

        # initialization
        self._sess.run(tf.global_variables_initializer())
        val_writer = tf.summary.FileWriter('logs', self._sess.graph)
        merged = tf.summary.merge_all()
        train_history = []  # to save iteration results
        x_all, z_all, p_all, w_all, y_all = [], [], [], [], []

        for step in range(self._config.num_iterations + 1):
            # Generate MC sample AS the training input
            dw_train, x_train = self.generate_feed()

            y, p, z, w = self._sess.run([self._train_ops, self.all_p, self.all_z, self.all_w],
                                        feed_dict={self._dw: dw_train,
                                                   self._x: x_train, self._is_training: True},
                                        )

            x_train_orig = np.reshape(x_train, [self._config.batch_size, self._config.dim,
                                                self._config.clayer, self._T + 1])
            x_all.append(x_train_orig)
            p_all.append(p)
            z_all.append(z)
            w_all.append(w)
            y_all.append(y)

            ### Validation Data Eval.
            self.validation_train(feed_dict_valid, train_history, merged, t0, step, val_writer)

        save_history(export_folder, train_history, x_all, z_all, p_all, w_all, y_all)
        return np.array(train_history)

    def build2(self):
        """"
           y : State
           dw : Brownian,  x : drift deterministic
           z : variance
        """
        t0 = time.time()
        TT = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
        M = self._config.dim  # Nb of assets

        ### dim X Ntime_interval for Stochastic Process
        self._dw = tf.placeholder(TF_DTYPE,
                                  [None, self._dim * self._config.clayer, self._T],
                                  name="dW")
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim * self._config.clayer,
                                            self._T + 1],
                                 name="X")

        ### Regularization
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.01, scope=None)

        ### Initialization
        ### x : state,  Cost
        self._y_init = tf.Variable(tf.random_uniform(
            [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
            dtype=TF_DTYPE), name='y_init')

        ## Control
        z_init = tf.Variable(
            tf.random_uniform([1, self._dim], minval=0.01, maxval=0.3, dtype=TF_DTYPE), name='z_init'
            # tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE)
        )

        p0 = tf.Variable(tf.random_uniform(shape=[self._config.batch_size],
                                           minval=0.0, maxval=0.0, dtype=TF_DTYPE), name='p_init')

        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE, name='all_ones')
        z0 = tf.matmul(all_one_vec, z_init)

        w0 = tf.Variable(tf.random.uniform([self._config.batch_size, M * self._config.clayer],
                                           minval=0.1, maxval=0.3, dtype=TF_DTYPE), name='w_init')

        all_p, all_z, all_w, all_y = [p0], [z0], [w0], []
        with tf.variable_scope("forward"):
            for t in range(1, self._T):
                # y = (y_old
                #        - self._bsde.delta_t * (self._bsde.f_tf(TT[t], self._x[:, :, t], y_old, z))
                #        + tf.reduce_sum(z * self._dw[:, :, t], 1, keepdims=True))

                ## Neural Network per Time Step, Calculate Gradient
                ######################################################################
                ## Z = [batch, dim * clayer] -> [batch, 4] for optionprice config
                if self._usemodel == 'lstm':
                    z = self.subnetwork.build([self._x[:, :, t - 1]], t - 1) / self._dim

                elif self._usemodel == 'ff':
                    z = self.subnetwork.build(self._x[:, :self._dim, t - 1], t - 1) / self._dim

                elif self._usemodel == 'attn':
                    z = self.subnetwork.build([self._x[:, :self._dim, t - 1]], t - 1) / self._dim

                elif self._usemodel == 'dila':
                    z = self.subnetwork.build([self._x[:, :, t - 1]], t - 1) / self._dim

                elif self._usemodel == 'biattn':
                    z = self.subnetwork.build(self._x[:, :, t - 1], t - 1) / self._dim
                all_z.append(z)

                ######################################################################
                y = z
                all_y.append(y)

                ######################################################################
                if t == 1:
                    w = 0.0 + z

                else:
                    a = 1
                    ### t=1 has issue
                    # w = 0.0 + z + 1 / tf.sqrt((tf.nn.moments(
                    #  tf.log((self._x[:, :M, 1:t ]) / (
                    #          self._x[:, :M, 0:t-1 ])), axes=2)[1] + self._smooth))

                # w = z
                # w = w / tf.reduce_sum(w, -1, keepdims=True)  ### Normalize Sum to 1
                # all_w.append(w)
                """
                  https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras
                
                
                def softargmax(x, beta=1e10):
                   x = tf.convert_to_tensor(x)
                   x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
                   return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
  
                
                """
                n_class =  self._dim  # n_class
                temperature = 3.0
                
                # Define weights for the log reg, n_classes = dim.
                W = tf.Variable(tf.zeros([self._dim, self._dim]), name='logregw')
                b = tf.Variable(tf.zeros([self._dim]), name='logregb')
                
                # let Y = Wx + b with softmax over classes
                w = tf.nn.softmax(tf.matmul(z, W) + b)

                class_label = [ [ 0.2, 0.2, 0.2 ]
                                [ 0.2, 0.2, 0.2 ]
                                [ 0.2, 0.2, 0.2 ]
                              ]
     
                class_label = tf.convert_to_tensor(class_label)

                ### Output is a tensor of shape (Nsample, :M, 1 )
                #### Its a kind of pseudo ArgMax differentiable....  
                w = tf.dot(w, class_label  )

                ######################################################################
                # p =  p_old * (1 + tf.reduce_sum( w * (self._x[:, :, t] / self._x[:, :, t-1] - 1), 1))
                # p = tf.reduce_sum(w * (self._x[:, :M, t] / self._x[:, :M, t - 1] - 1), 1)
                p = tf.reduce_sum(w * (self._x[:, :M, t]), 1)

                all_p.append(p)

            # Terminal time
            # y = (y  - self._bsde.delta_t * self._bsde.f_tf(TT[-1], self._x[:, :, -2], y, z)
            #        + tf.reduce_sum(z * self._dw[:, :, -1], 1, keepdims=True) )
            # y = self._x[:, :, -2]
            # all_y.append(y)

            self.all_y = tf.stack(all_y, axis=-1)
            self.all_z = tf.stack(all_z, axis=-1)
            self.all_w = tf.stack(all_w, axis=-1)
            self.all_p = tf.stack(all_p, axis=-1)
            p = self.all_p

            # Final Difference :
            #######  -Sum(ri)   +Sum(ri**2)
            # delta =  -tf.reduce_sum(p[:, 1:], 1) + tf.nn.moments(p[:, 1:], axes=1)[1]*10.0
            delta = tf.nn.moments(p[:, 1:10], axes=1)[1] * 10000.0 + tf.nn.moments(p[:, 10:20], axes=1)[1] * 10000.0 + \
                    tf.nn.moments(p[:, 20:], axes=1)[1] * 10000.0

            weights = tf.trainable_variables()  # all vars of your graph
            reg_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            self._loss = tf.reduce_mean(delta)
            self._loss = self._loss + reg_penalty  # this loss needs to be minimized

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
        self._dw = tf.placeholder(TF_DTYPE,
                                  [None, self._config.clayer * self._dim, self._T],
                                  name="dW")
        self._x = tf.placeholder(TF_DTYPE, [None, self._config.clayer * self._dim,
                                            self._T + 1],
                                 name="X")

        ### Initialization
        ## x : state,  Cost
        self._y_init = tf.Variable(
            tf.random_uniform(
                [1], minval=self._config.y_init_range[0], maxval=self._config.y_init_range[1],
                dtype=TF_DTYPE, name='y_iniy'
            )
        )

        ## Control
        z_init = tf.Variable(
            tf.random_uniform([1, self._dim], minval=-0.1, maxval=0.1, dtype=TF_DTYPE), name='z_init')
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE, name='all_ones')
        y = all_one_vec * self._y_init
        z = tf.matmul(all_one_vec, z_init)

        with tf.variable_scope("forward"):
            for t in range(0, self._T - 1):
                y = (
                        y
                        - self._bsde.delta_t * (
                            self._bsde.f_tf(TT[t], self._x[:, :self._dim, t], y, z))
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

    def sample_files(self, X, dW, idx):
        bs = self._config.batch_size
        dw, x = dW[bs * idx:bs * (idx + 1), ...], X[bs * idx:bs * (idx + 1), ...]

        dw_valid = np.reshape(dw,
                              [self._config.batch_size,
                               self._config.clayer * self._dim, self._T])

        x_valid = np.reshape(x,
                             [self._config.batch_size,
                              self._config.clayer * self._dim, self._T + 1])

        return dw_valid, x_valid

    def predict_sequence(self, x_file_path, dw_file_path):
        saver = tf.train.Saver()
        save_path = os.path.join('ckpt', 'model.ckpt')
        x_all, dw_all = np.load(x_file_path), np.load(dw_file_path)
        n = x_all.shape[0]
        saver.restore(self._sess, save_path)
        p_all, z_all, w_all = [], [], []
        print("Predicting over the sequence, results will be saved as np array...")
        for idx in range(n // self._config.batch_size):
            dw, x = self.sample_files(x_all, dw_all, idx)
            p, z, w = self._sess.run([self.all_p, self.all_z, self.all_w],
                                     feed_dict={self._dw: dw,
                                                self._x: x, self._is_training: False})
            p_all.append(p)
            z_all.append(z)
            w_all.append(w)

        p_all = np.concatenate(p_all, axis=0)
        z_all = np.concatenate(z_all, axis=0)
        w_all = np.concatenate(w_all, axis=0)
        np.save('logs/p.npy', p_all)
        np.save('logs/z.npy', z_all)
        np.save('logs/w.npy', w_all)
