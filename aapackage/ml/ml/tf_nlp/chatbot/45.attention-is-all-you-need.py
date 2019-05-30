#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import os
import re
import time

import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf

# In[2]:


def build_dataset(words, n_words, atleast=1):
    count = [["PAD", 0], ["GO", 1], ["EOS", 2], ["UNK", 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# In[3]:


lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conv_lines = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

convs = []
for line in conv_lines[:-1]:
    _line = line.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_line.split(","))

questions = []
answers = []

for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i + 1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return " ".join([i.strip() for i in filter(None, text.split())])


clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]


# In[4]:


concat_from = " ".join(short_questions + question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(
    concat_from, vocabulary_size_from
)
print("vocab from size: %d" % (vocabulary_size_from))
print("Most common words", count_from[4:10])
print("Sample data", data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print("filtered vocab size:", len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from) / vocabulary_size_from, 4) * 100))


# In[5]:


concat_to = " ".join(short_answers + answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print("vocab from size: %d" % (vocabulary_size_to))
print("Most common words", count_to[4:10])
print("Sample data", data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print("filtered vocab size:", len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to) / vocabulary_size_to, 4) * 100))


# In[6]:


GO = dictionary_from["GO"]
PAD = dictionary_from["PAD"]
EOS = dictionary_from["EOS"]
UNK = dictionary_from["UNK"]


# In[7]:


for i in range(len(short_answers)):
    short_answers[i] += " EOS"


# In[8]:


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable("gamma", params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable("beta", params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):

    T_q = tf.shape(queries)[1]
    T_k = tf.shape(keys)[1]

    Q = tf.layers.dense(queries, num_units, name="Q")
    K_V = tf.layers.dense(keys, 2 * num_units, name="K_V")
    K, V = tf.split(K_V, 2, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    align = align / np.sqrt(K_.get_shape().as_list()[-1])

    paddings = tf.fill(tf.shape(align), float("-inf"))

    key_masks = k_masks
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])
    align = tf.where(tf.equal(key_masks, 0), paddings, align)

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])
        lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    query_masks = tf.to_float(q_masks)
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])
    align *= query_masks

    outputs = tf.matmul(align, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    outputs += queries
    outputs = layer_norm(outputs)
    return outputs


def pointwise_feedforward(inputs, hidden_units, activation=None):
    outputs = tf.layers.dense(inputs, 4 * hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


def learned_position_encoding(inputs, mask, embed_dim):
    T = tf.shape(inputs)[1]
    outputs = tf.range(tf.shape(inputs)[1])  # (T_q)
    outputs = tf.expand_dims(outputs, 0)  # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])  # (N, T_q)
    outputs = embed_seq(outputs, T, embed_dim, zero_pad=False, scale=False)
    return tf.expand_dims(tf.to_float(mask), -1) * outputs


def sinusoidal_position_encoding(inputs, mask, repr_dim):
    T = tf.shape(inputs)[1]
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / C)


class Chatbot:
    def __init__(
        self,
        size_layer,
        embedded_size,
        from_dict_size,
        to_dict_size,
        learning_rate,
        num_blocks=2,
        num_heads=8,
        min_freq=50,
    ):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]

        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))

        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        def forward(x, y):
            encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, x)
            en_masks = tf.sign(x)
            encoder_embedded += sinusoidal_position_encoding(x, en_masks, embedded_size)

            for i in range(num_blocks):
                with tf.variable_scope("encoder_self_attn_%d" % i, reuse=tf.AUTO_REUSE):
                    encoder_embedded = multihead_attn(
                        queries=encoder_embedded,
                        keys=encoder_embedded,
                        q_masks=en_masks,
                        k_masks=en_masks,
                        future_binding=False,
                        num_units=size_layer,
                        num_heads=num_heads,
                    )

                with tf.variable_scope("encoder_feedforward_%d" % i, reuse=tf.AUTO_REUSE):
                    encoder_embedded = pointwise_feedforward(
                        encoder_embedded, embedded_size, activation=tf.nn.relu
                    )

            decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, y)
            de_masks = tf.sign(y)
            decoder_embedded += sinusoidal_position_encoding(y, de_masks, embedded_size)

            for i in range(num_blocks):
                with tf.variable_scope("decoder_self_attn_%d" % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(
                        queries=decoder_embedded,
                        keys=decoder_embedded,
                        q_masks=de_masks,
                        k_masks=de_masks,
                        future_binding=True,
                        num_units=size_layer,
                        num_heads=num_heads,
                    )

                with tf.variable_scope("decoder_attn_%d" % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(
                        queries=decoder_embedded,
                        keys=encoder_embedded,
                        q_masks=de_masks,
                        k_masks=en_masks,
                        future_binding=False,
                        num_units=size_layer,
                        num_heads=num_heads,
                    )

                with tf.variable_scope("decoder_feedforward_%d" % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = pointwise_feedforward(
                        decoder_embedded, embedded_size, activation=tf.nn.relu
                    )

            return tf.layers.dense(decoder_embedded, to_dict_size, reuse=tf.AUTO_REUSE)

        self.training_logits = forward(self.X, decoder_input)

        def cond(i, y, temp):
            return i < 2 * tf.reduce_max(self.X_seq_len)

        def body(i, y, temp):
            logits = forward(self.X, y)
            ids = tf.argmax(logits, -1)[:, i]
            ids = tf.expand_dims(ids, -1)
            temp = tf.concat([temp[:, 1:], ids], -1)
            y = tf.concat([temp[:, -(i + 1) :], temp[:, : -(i + 1)]], -1)
            y = tf.reshape(y, [tf.shape(temp)[0], 2 * tf.reduce_max(self.X_seq_len)])
            i += 1
            return i, y, temp

        target = tf.fill([batch_size, 2 * tf.reduce_max(self.X_seq_len)], GO)
        target = tf.cast(target, tf.int64)
        self.target = target

        _, self.predicting_ids, _ = tf.while_loop(cond, body, [tf.constant(0), target, target])

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.Y, weights=masks
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[9]:


embedded_size = 256
learning_rate = 0.001
batch_size = 16
epoch = 20


# In[10]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(
    embedded_size, embedded_size, len(dictionary_from), len(dictionary_to), learning_rate
)
sess.run(tf.global_variables_initializer())


# In[11]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


# In[12]:


X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)


# In[13]:


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


# In[14]:


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    for k in range(0, len(short_questions), batch_size):
        index = min(k + batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch(X[k:index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k:index], PAD)
        predicted, accuracy, loss, _ = sess.run(
            [model.predicting_ids, model.accuracy, model.cost, model.optimizer],
            feed_dict={model.X: batch_x, model.Y: batch_y},
        )
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= len(short_questions) / batch_size
    total_accuracy /= len(short_questions) / batch_size
    print(predicted)
    print("epoch: %d, avg loss: %f, avg accuracy: %f\n" % (i + 1, total_loss, total_accuracy))


# In[15]:


for i in range(len(batch_x)):
    print("row %d" % (i + 1))
    print(
        "QUESTION:", " ".join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]])
    )
    print(
        "REAL ANSWER:",
        " ".join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]),
    )
    print(
        "PREDICTED ANSWER:",
        " ".join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
        "\n",
    )


# In[16]:


batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD)
predicted = sess.run(model.predicting_ids, feed_dict={model.X: batch_x})

for i in range(len(batch_x)):
    print("row %d" % (i + 1))
    print(
        "QUESTION:", " ".join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]])
    )
    print(
        "REAL ANSWER:",
        " ".join([rev_dictionary_to[n] for n in batch_y[i] if n not in [0, 1, 2, 3]]),
    )
    print(
        "PREDICTED ANSWER:",
        " ".join([rev_dictionary_to[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
        "\n",
    )


# In[ ]:
