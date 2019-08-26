# -*- coding: utf-8 -*-
"""

https://blog.evjang.com/2016/11/tutorial-categorical-variational.html


"""

import tensorflow as tf


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



def get_class_label(k) :
  class_label = np.array([ [ 0.2, 0.3, 0.4 ],
                [ 0.1, 0.5, 0.1 ],
                [ 0.6, 0.2, 0.6 ],
              ])

  return tf.convert_to_tensor(class_label[k])



def argmax_smooth(z, temperature, hard=True):
    """


    """
    ### Proba smoothed by temperature
    z_proba = gumbel_softmax_sample(z, temperature)

    ### Get hard labels
    z_hard = get_class_label(tf.argmax(y_proba,1) )
    z_hard = tf.cast( z_hard, z_proba.dtype)

    w_new = tf.stop_gradient(z_hard - z_proba) + z_proba
    return w_new






                n_class =  self._dim  # n_class
                temperature = 10.0
                beta = 1.0 / temperature


                def softargmax(x, beta=1e10):
                   x = tf.convert_to_tensor(x)
                   x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
                   return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)


                #  z is already sigmoid per dim.
                #  We need to flatten / peak the softmax with beta = 1/temperature
                z = softargmax(z, beta=1e10)

                """
                # Define weights for the log reg, n_classes = dim.
                W = tf.Variable(tf.zeros([self._dim, n_class ]), name='logregw')
                b = tf.Variable(tf.zeros([ n_class ]), name='logregb')
                # let Y = Wx + b with softmax over classes
                w = tf.nn.softmax(tf.matmul(z, W) + b)
                """

                ##### n_class X label  ####################################################
                class_label =  tf.convert_to_tensor( np.array([ [ 0.2, 0.3, 0.4 ],
                                       [ 0.1, 0.5, 0.1 ],
                                       [ 0.6, 0.2, 0.6 ],
                              ]) )

                ##### Not clear how to Vectorize this part in TF  ???
                batch_size = 64
                for ii in range( batch_size) :
                   ### Calculate : Vector = Sum( Proba_i * class_label[i,:] ) for 1 sample
                   wtmp = np.diag(z[ ii, :self.dim])  # 3x3 shape
                   wtmp =  np.sum( np.dot( class_label, wtmp ) , axis=0)  # 1 x self.dim
                   w[ii, :] = tf.convert_to_tensor( wtmp )


                """ TF pseudo version
                   Use : ?????
                       https://stackoverflow.com/questions/48626610/for-loop-in-a-tensor-in-tensorflow
                   wtmp = tf.diag(z[ ii, :self.dim])  # 3x3 shape
                   wtmp =  tf.reduce_sum( tf.mult( class_label, wtmp ) , axis=0)  # 1 x self.dim
                   w[ii, :] = wtmp                        
                """


                """  NUMPY version working, check purpose
                  n_class = 3
                  ndim
                  i = 5
                  w = np.zeros((500, 6 ))
                  w[:, :3 ] = np.array([ 0.2, 0.3, 0.5 ])
                  #### Sum( Proba_i * labe_i)
                  wtmp = np.diag(w[ i, :self.dim])
                  wtmp =  np.sum( np.dot(wtmp, class_label ) , axis=0)
                  #   array([0.37, 0.31, 0.41])
                """











