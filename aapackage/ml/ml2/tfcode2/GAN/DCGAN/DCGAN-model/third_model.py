import tensorflow as tf

Z = tf.placeholder(tf.float32, shape = [1, 100])
G_W_1 = tf.Variable(tf.random_normal([100, 32 * 8 * 4 * 4], stddev = 0.5))
G_b_1 = tf.Variable(tf.random_normal([32 * 8 * 4 * 4], stddev = 0.01))

G_W_2 = tf.Variable(tf.random_normal([5, 5, 128, 256], stddev = 0.5))
G_b_2 = tf.Variable(tf.random_normal([256], stddev = 0.01))

G_W_3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev = 0.5))
G_b_3 = tf.Variable(tf.random_normal([128], stddev = 0.01))

G_W_4 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = 0.5))
G_b_4 = tf.Variable(tf.random_normal([32], stddev = 0.01))

G_W_5 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev = 0.5))
G_b_5 = tf.Variable(tf.random_normal([16], stddev = 0.01))

G_W_6 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev = 0.5))
G_b_6 = tf.Variable(tf.random_normal([3], stddev = 0.01))
