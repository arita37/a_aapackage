#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os


# In[2]:


def build_dataset(words, n_words, atleast=1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
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


lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
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
    return ' '.join([i.strip() for i in filter(None, text.split())])

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


concat_from = ' '.join(short_questions+question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('filtered vocab size:',len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from)/vocabulary_size_from,4)*100))


# In[5]:


concat_to = ' '.join(short_answers+answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('vocab from size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('filtered vocab size:',len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to)/vocabulary_size_to,4)*100))


# In[6]:


GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']


# In[7]:


for i in range(len(short_answers)):
    short_answers[i] += ' EOS'


# In[8]:


class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size, 
                 from_dict_size, to_dict_size, batch_size,
                 grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5):
        
        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(),reuse=reuse),
                input_keep_prob=0.8,
                output_keep_prob=0.8,
                state_keep_prob=0.8)
        
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        self.encoder_out = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        
        def bahdanau(size):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = size, 
                                                                    memory = self.encoder_out)
            return tf.contrib.seq2seq.AttentionWrapper(cell = cells(size), 
                                                        attention_mechanism = attention_mechanism,
                                                        attention_layer_size = size)
        
        def luong(size):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = size, 
                                                                    memory = self.encoder_out)
            return tf.contrib.seq2seq.AttentionWrapper(cell = cells(size), 
                                                        attention_mechanism = attention_mechanism,
                                                        attention_layer_size = size)
        
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = bahdanau(size_layer//2),
                cell_bw = luong(size_layer//2),
                inputs = self.encoder_out,
                sequence_length = self.X_seq_len,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_%d'%(n))
            encoder_embedded = tf.concat((out_fw, out_bw), 2)
            
        bi_state_c = tf.concat((state_fw[0].c, state_bw[0].c), -1)
        bi_state_h = tf.concat((state_fw[0].h, state_bw[0].h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        encoder_state = tuple([bi_lstm_state] * num_layers)
        
        dense = tf.layers.Dense(to_dict_size)
        
        with tf.variable_scope('decode'):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = size_layer, 
                memory = self.encoder_out,
                memory_sequence_length = self.X_seq_len)
            luong_cells = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = size_layer, 
                memory = self.encoder_out,
                memory_sequence_length = self.X_seq_len)
            bahdanau_cells = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            decoder_cells = tf.nn.rnn_cell.MultiRNNCell([luong_cells, bahdanau_cells])
            main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs = tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
                sequence_length = self.Y_seq_len,
                embedding = decoder_embeddings,
                sampling_probability = 1 - force_teaching_ratio,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cells,
                helper = training_helper,
                initial_state = decoder_cells.zero_state(batch_size, tf.float32),
                output_layer = tf.layers.Dense(to_dict_size))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
            
        with tf.variable_scope('decode', reuse=True):
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, beam_width)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = size_layer, 
                memory = encoder_out_tiled,
                memory_sequence_length = X_seq_len_tiled)
            luong_cells = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer,reuse=True) for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = size_layer, 
                memory = encoder_out_tiled,
                memory_sequence_length = X_seq_len_tiled)
            bahdanau_cells = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer,reuse=True) for _ in range(num_layers)]),
                attention_mechanism = attention_mechanism,
                attention_layer_size = size_layer)
            decoder_cells = tf.nn.rnn_cell.MultiRNNCell([luong_cells, bahdanau_cells])
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = decoder_cells,
                embedding = decoder_embeddings,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS,
                initial_state = decoder_cells.zero_state(batch_size * beam_width, tf.float32),
                beam_width = beam_width,
                output_layer = tf.layers.Dense(to_dict_size, _reuse=True),
                length_penalty_weight = 0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = False,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]
        
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        l2 = sum(1e-5 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[9]:


size_layer = 512
num_layers = 2
embedded_size = 256
learning_rate = 1e-2
batch_size = 16
epoch = 30


# In[10]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, len(dictionary_from), 
                len(dictionary_to), batch_size, learning_rate)
sess.run(tf.global_variables_initializer())


# In[11]:


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
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
        index = min(k+batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch(X[k: index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
        predicted, accuracy,loss, _ = sess.run([model.predicting_ids, 
                                                model.accuracy, model.cost, model.optimizer], 
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y})
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= (len(short_questions) / batch_size)
    total_accuracy /= (len(short_questions) / batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))


# In[15]:


for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')


# In[16]:


batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD)
predicted = sess.run(model.predicting_ids, feed_dict={model.X:batch_x})

for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')


# In[ ]:




