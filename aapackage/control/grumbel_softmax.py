# -*- coding: utf-8 -*-
"""
Grumbel Softmax 
as Differential Argmax

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.set_random_seed(42)



BATCHS_IN_EPOCH = 100
BATCH_SIZE = 10
EPOCHS = 200  # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
GENERATOR_TRAINING_FACTOR = 10  # for every training of the disctiminator we'll train the generator 10 times
LEARNING_RATE = 0.0007
TEMPERATURE = 0.001  # we use a constant, but for harder problems we should




number_to_prob = {
    0: 0.0,
    1: 0.0,
    2: 0.1,
    3: 0.3,
    4: 0.6
}


def generate_text():
    while True:
        yield np.random.choice(number_to_prob.keys(), p=number_to_prob.values(), size=1)


dataset = tf.data.Dataset.from_generator(generate_text,
                                         output_types=tf.int32,
                                         output_shapes=1).batch(BATCH_SIZE)
value = dataset.make_one_shot_iterator().get_next()
value = tf.one_hot(value, len(number_to_prob))
value = tf.squeeze(value, axis=1)


#### Generate of discrete sequence using Grumbel Distribition
def generator():
    with tf.variable_scope('generator'):
        logits = tf.get_variable('logits', initializer=tf.ones([len(number_to_prob)]))
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(TEMPERATURE, logits=logits)
        probs = tf.nn.softmax(logits)
        generated = gumbel_dist.sample(BATCH_SIZE)
        return generated, probs




def discriminator(x):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.fully_connected(x,
                                                 num_outputs=1,
                                                 activation_fn=None)







generated_outputs, generated_probs = generator()
discriminated_real = discriminator(value)
discriminated_generated = discriminator(generated_outputs)

d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_real,
                                            labels=tf.ones_like(discriminated_real)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.zeros_like(discriminated_generated)))
d_loss = d_loss_real + d_loss_fake


g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.ones_like(discriminated_generated)))

all_vars = tf.trainable_variables()
g_vars = [var for var in all_vars if var.name.startswith('generator')]
d_vars = [var for var in all_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)




####### Train Generator/ discriminator
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    learned_probs = []
    for _ in range(1000):
        for _ in range(BATCHS_IN_EPOCH):
            sess.run(d_train_opt)
            for _ in range(GENERATOR_TRAINING_FACTOR):
                sess.run(g_train_opt)
        learned_probs.append(sess.run(generated_probs))






plt.figure(figsize=(10, 2))
prob_errors = [np.array(learned_prob) - np.array(number_to_prob.values())
               for learned_prob in learned_probs]
plt.imshow(np.transpose(prob_errors),
           cmap='bwr',
           aspect='auto',
           vmin=-2,
           vmax=2)
plt.xlabel('epoch')
plt.ylabel('number')
plt.colorbar(aspect=10, ticks=[-2, 0, 2]);





####################################################################################################
####################################################################################################
class_label = np.array([ [ 0.2, 0.3, 0.4 ],
                [ 0.1, 0.5, 0.1 ],
                [ 0.6, 0.2, 0.6 ],
              ])

### Output is a tensor of shape (Nsample, :M, 1 ), same than z
#### Its a kind of pseudo ArgMax differentiable....
### Average Label from proba
wtemp = [0] * n_class
for wi in w :
    for label_j in class_label:
      wtemp = wtemp + wi*label_j



####################################################################################################
####################################################################################################
n_class = 3
ndim = 5
i = 5

w = np.zeros((500, 6 ))
w[:, :3 ] = np.array([ 0.2, 0.3, 0.5 ])


########## data setup
### Not clear how to make diagnonal matrix from vector in TF
wtmp = np.diag(w[ i, :self.dim])

#iden  = np.identity(n_class)
#wtmp = np.dot(iden, wtmp.reshape(1,-1) )


#### Sum( Proba_i * labe_i)
wtmp =  np.sum( np.dot(wtmp, class_label ) , axis=0)
"""
np.sum( label_final , axis=0)
   array([0.37, 0.31, 0.41])


"""

####################################################################################################
####################################################################################################
wtmp = w[0, :3 ]

sess = tf.compat.v1.InteractiveSession()

class_label = tf.convert_to_tensor( np.array([ [ 1, 2, 3 ],
                [ 40, 50 , 60 ],
                [ 70.0, 80, 90 ],
              ]) )


temperature = 1.0
z = tf.convert_to_tensor( [ 0.2, 0.5, 0.3 ] , dtype= tf.float64)
z = tf.nn.softmax( z / temperature)
z.eval()


w = tf.linalg.diag(z)
w = tf.expand_dims(w, axis=0)


w1 = tf.matmul( w, tf.expand_dims(class_label, axis=0) )
w2 = tf.reduce_sum( w1 , axis=1)

w2.eval()
w.eval()




def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)










####################################################################################################
####################################################################################################


def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y








####################################################################################################
####################################################################################################
"""
### Output is a tensor of shape (Nsample, :M, 1 ), same than z
#### Its a kind of pseudo ArgMax differentiable....
### Average Label from proba
wtemp = [0] * n_class
for wi in w :
    for label_j in class_label:
      wtemp = wtemp + wi*label_j


wtemp = tf.convert_to_tensor(wtemp)

w = wtemp   #Tensor od dim (Nsample, :M)
"""















