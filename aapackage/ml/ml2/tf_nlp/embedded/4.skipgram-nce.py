#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import tensorflow as tf
from collections import Counter
from sklearn.cross_validation import train_test_split
import time


# In[2]:


trainset = sklearn.datasets.load_files(container_path = 'data', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset,1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# In[3]:


texts = ' '.join(trainset.data)
words = texts.split()
word2freq = Counter(words)
print("Total words:", len(words))


# In[4]:


_words = set(words)
word2idx = {c: i for i, c in enumerate(_words)}
idx2word = {i: c for i, c in enumerate(_words)}
vocab_size = len(idx2word)
indexed = [word2idx[w] for w in words]
print('Vocabulary size:', vocab_size)


# In[15]:


class SKIPGRAM:
    def __init__(self, sample_size, vocab_size, embedded_size, window_size=3):
        self.X = tf.placeholder(tf.int32, shape=[None])
        self.Y = tf.placeholder(tf.int32, shape=[None, 1])
        self.embedding = tf.Variable(tf.truncated_normal([vocab_size, embedded_size],
                                                      stddev=1.0 / np.sqrt(embedded_size)))
        self.bias = tf.Variable(tf.zeros([vocab_size]))
        embedded = tf.nn.embedding_lookup(self.embedding, self.X)
        self.cost = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.embedding,
            biases=self.bias,
            labels=self.Y,
            inputs=embedded,
            num_sampled=sample_size,
            num_classes=vocab_size))
        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost)
        self.valid_dataset = tf.placeholder(tf.int32, shape=[None])
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
        normalized_embeddings = self.embedding / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


# In[6]:


batch_size = 128
embedded_size = 128
skip_window = 5
epoch = 10
valid_size = 10
nearest_neighbors = 8


# In[16]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = SKIPGRAM(batch_size,vocab_size,embedded_size)
sess.run(tf.global_variables_initializer())


# In[10]:


def get_y(words, idx):
    skipped = np.random.randint(1, skip_window+1)
    left = idx - skip_window if (idx - skipped) > 0 else 0
    right = idx + skipped
    y = words[left: idx] + words[idx+1: right+1]
    return list(set(y))

def make_xy(int_words):
    x,y = [], []
    for i in range(0, len(int_words)):
        input_w = int_words[i]
        labels = get_y(int_words, i)
        x.extend([input_w] * len(labels))
        y.extend(labels)
    return np.array(x), np.array(y)


# In[11]:


X, Y = make_xy(indexed)


# In[17]:


for i in range(epoch):
    total_cost = 0
    for k in range(0,(X.shape[0] // batch_size) * batch_size,batch_size):
        batch_x = X[k:k+batch_size]
        batch_y = Y[k:k+batch_size,np.newaxis]
        cost,_ = sess.run([model.cost,model.optimizer],feed_dict={model.X:batch_x,
                                                                 model.Y:batch_y})
        total_cost += cost
    total_cost /= (X.shape[0] // batch_size)
    print('epoch %d, avg loss %f'%(i+1,total_cost))
    random_valid_size = np.random.choice(indexed, valid_size)
    similarity = sess.run(model.similarity,feed_dict={model.valid_dataset:random_valid_size})
    for no, i in enumerate(random_valid_size):
        valid_word = idx2word[i]
        nearest = (-similarity[no, :]).argsort()[1:nearest_neighbors + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(nearest_neighbors):
            close_word = idx2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)


# In[ ]:




