import logging
import time
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float32
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


def log(s):
    logging.info(s)


class subnetwork_bidirectionalattn(object):

    def __init__(self, config):
        self._config = config

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self._config.n_hidden_lstm, state_is_tuple=False)

    def build(self, x, i):
        with tf.variable_scope("Global_RNN", reuse=i > 0):
            x = tf.expand_dims(x, 1)
            backward_rnn_cells = self.lstm_cell()#tf.nn.rnn_cell.MultiRNNCell(
            #    [self.lstm_cell() for _ in range(len(self._config.num_hiddens))], state_is_tuple=False
            #)
            forward_rnn_cells = self.lstm_cell() #tf.nn.rnn_cell.MultiRNNCell(
            #    [self.lstm_cell() for _ in range(len(self._config.num_hiddens))], state_is_tuple=False
            #)

            outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
                forward_rnn_cells,
                backward_rnn_cells,
                x,
                dtype=TF_DTYPE,
            )

            outputs = list(outputs)
            attention_w = tf.get_variable("attention_v1", [self._config.n_hidden_lstm], TF_DTYPE)
            query = tf.layers.dense(tf.expand_dims(last_state[0][:, self._config.n_hidden_lstm:], 1),
                                    self._config.n_hidden_lstm)
            keys = tf.layers.dense(outputs[0], self._config.n_hidden_lstm)
            align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
            align = tf.nn.tanh(align)
            outputs[0] = tf.squeeze(
                tf.matmul(tf.transpose(outputs[0], [0, 2, 1]), tf.expand_dims(align, 2)), 2
            )
            outputs[0] = tf.concat([outputs[0], last_state[0][:, self._config.n_hidden_lstm:]], 1)

            attention_w = tf.get_variable("attention_v2", [self._config.n_hidden_lstm], TF_DTYPE)
            query = tf.layers.dense(tf.expand_dims(last_state[1][:, self._config.n_hidden_lstm:], 1),
                                    self._config.n_hidden_lstm)
            keys = tf.layers.dense(outputs[1], self._config.n_hidden_lstm)
            align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
            align = tf.nn.tanh(align)
            outputs[1] = tf.squeeze(
                tf.matmul(tf.transpose(outputs[1], [0, 2, 1]), tf.expand_dims(align, 2)), 2
            )
            outputs[1] = tf.concat([outputs[1], last_state[1][:, self._config.n_hidden_lstm:]], 1)

            with tf.variable_scope("decoder", reuse=i > 0):
                self.backward_rnn_cells_dec = self.lstm_cell()#tf.nn.rnn_cell.MultiRNNCell(
                #   [self.lstm_cell() for _ in range(len(self._config.num_hiddens))], state_is_tuple=False
                #)
                self.forward_rnn_cells_dec = self.lstm_cell()#tf.nn.rnn_cell.MultiRNNCell(
                #    [self.lstm_cell() for _ in range(len(self._config.num_hiddens))], state_is_tuple=False
                #)
                self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(
                    self.forward_rnn_cells_dec,
                    self.backward_rnn_cells_dec,
                    x,
                    initial_state_fw=outputs[0],
                    initial_state_bw=outputs[1],
                    dtype=TF_DTYPE,
                )
            self.outputs = list(self.outputs)
            attention_w = tf.get_variable("attention_v3", [self._config.n_hidden_lstm], TF_DTYPE)
            query = tf.layers.dense(
                tf.expand_dims(self.last_state[0][:, self._config.n_hidden_lstm:], 1), self._config.n_hidden_lstm
            )
            keys = tf.layers.dense(self.outputs[0], self._config.n_hidden_lstm)
            align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
            align = tf.nn.tanh(align)
            self.outputs[0] = tf.squeeze(
                tf.matmul(tf.transpose(self.outputs[0], [0, 2, 1]), tf.expand_dims(align, 2)), 2
            )

            attention_w = tf.get_variable("attention_v4", [self._config.n_hidden_lstm], TF_DTYPE)
            query = tf.layers.dense(
                tf.expand_dims(self.last_state[1][:, self._config.n_hidden_lstm:], 1), self._config.n_hidden_lstm
            )
            keys = tf.layers.dense(self.outputs[1], self._config.n_hidden_lstm)
            align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
            align = tf.nn.tanh(align)
            self.outputs[1] = tf.squeeze(
                tf.matmul(tf.transpose(self.outputs[1], [0, 2, 1]), tf.expand_dims(align, 2)), 2
            )
            self.outputs = tf.concat(self.outputs, 1)
            self.logits = tf.layers.dense(self.outputs, self._config.dim)

            return self.logits


class subnetwork_dila(object):
    def __init__(self, config):
        self._config = config

    def rnn_reformat(self, x, input_dims, n_steps):
        x_ = tf.transpose(x, [1, 0, 2])
        x_ = tf.reshape(x_, [-1, input_dims])
        return tf.split(x_, n_steps, 0)

    def contruct_cells(self, hidden_structs):
        cells = []
        for hidden_dims in hidden_structs:
            cells.append(tf.nn.rnn_cell.LSTMCell(hidden_dims, state_is_tuple=False, dtype=TF_DTYPE))
        return cells

    def dilated_rnn(self, cell, inputs, rate, scope="default"):
        n_steps = len(inputs)
        if not (n_steps % rate) == 0:
            zero_tensor = tf.zeros_like(inputs[0], dtype=TF_DTYPE)
            dilated_n_steps = n_steps // rate + 1
            for i_pad in range(dilated_n_steps * rate - n_steps):
                inputs.append(zero_tensor)
        else:
            dilated_n_steps = n_steps // rate
        dilated_inputs = [
            tf.concat(inputs[i * rate: (i + 1) * rate], axis=0) for i in range(dilated_n_steps)]
        dilated_outputs, states = tf.contrib.rnn.static_rnn(
            cell, dilated_inputs, dtype=TF_DTYPE, scope=scope
        )
        splitted_outputs = [tf.split(output, rate, axis=0) for output in dilated_outputs]
        unrolled_outputs = [output for sublist in splitted_outputs for output in sublist]
        return unrolled_outputs[:n_steps], states

    def multi_dilated_rnn(self, cells, inputs, dilations):
        x = copy.copy(inputs)
        count = 0
        for cell, dilation in zip(cells, dilations):
            x, states = self.dilated_rnn(cell, x, dilation, scope="multi_dilated_rnn_%d" % count)
            count += 1
        return x, states

    def build(self, x_reformat, t):
        with tf.variable_scope('Global_RNN', reuse=t > 0):
            hidden_structs = self._config.num_hiddens
            # x_reformat = self.rnn_reformat(x, self._config.dim, self._config.num_time_interval)
            cells = self.contruct_cells(hidden_structs)
            layer_outputs, self.last_state = self.multi_dilated_rnn(
                cells, x_reformat, self._config.dilations)
            weights = tf.Variable(
                tf.random_normal(shape=[hidden_structs[-1], self._config.num_hiddens[-1]], dtype=TF_DTYPE),
                dtype=TF_DTYPE)
            bias = tf.Variable(tf.random_normal(shape=[self._config.num_hiddens[-1]], dtype=TF_DTYPE),
                               dtype=TF_DTYPE)
            self.logits = tf.matmul(layer_outputs[-1], weights) + bias
            return self.logits


###############  LSTM
class subnetwork_lstm(object):
    def __init__(self, config):
        self._config = config

    def build(self, x, i):
        with tf.variable_scope('Global_RNN', reuse=i > 0):
            lstm = tf.nn.rnn_cell.LSTMCell(self._config.n_hidden_lstm, name='lstm_cell', reuse=i > 0)
            x, s = tf.nn.static_rnn(lstm, x, dtype=TF_DTYPE)
            x = tf.layers.dense(x[-1], self._config.num_hiddens[-1], name='dense_out', reuse=i > 0)
            return x


############### LSTM ATTN
class subnetwork_lstm_attn(object):
    def __init__(self, config):
        self._config = config

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self._config.n_hidden_lstm)

    def build(self, x, i):
        with tf.variable_scope('Global_RNN', reuse=i > 0):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self._config.n_hidden_lstm,
                memory=tf.expand_dims(x[0], axis=1),
                dtype=TF_DTYPE)

            self.rnn_cells = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.LSTMCell(self._config.n_hidden_lstm,
                                             name='lstm_cell',
                                             reuse=i > 0),
                attention_mechanism=attention_mechanism)

            self.outputs, self.last_state = tf.nn.static_rnn(self.rnn_cells, x, dtype=TF_DTYPE)
            self.out = tf.layers.dense(self.outputs[-1],
                                       self._config.num_hiddens[-1],
                                       name='dense_out', reuse=i > 0)
            return self.out


##############  FeedForward
class subnetwork_ff(object):
    """
      FeedForward network, connected by Input.
      Final layer no Activation function,
      dim : nb of assets.
      num_hiddens = [dim, dim+10, dim+10, dim]
    """

    def __init__(self, config, is_training):
        self._config = config
        # ops for statistics update of batch normalization
        self._extra_train_ops = []
        self._is_training = is_training

    def build(self, x, i):
        # self.x = x
        name = str(i+1)
        with tf.variable_scope(name):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = tf.layers.batch_normalization(x, training=self._is_training)
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

    def _dense_batch_layer(self, input_, output_size, activation_fn=None, stddev=5.0, name="linear"):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable(
                "Matrix",
                [shape[1], output_size],
                TF_DTYPE,
                tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size)),
            )
            hiddens = tf.matmul(input_, weight)
            # hiddens_bn = self._batch_norm(hiddens)
            hiddens_bn = tf.layers.batch_normalization(hiddens, training=self._is_training)
        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn
