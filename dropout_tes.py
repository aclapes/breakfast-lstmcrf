import numpy as np
import tensorflow as tf

np.set_printoptions(3)

drop_rate = 0.3

batch_size = 3
num_steps = 4
num_features = 10

X = np.random.random((batch_size,num_steps,num_features))
Xtf = tf.constant(X)

def variational_dropout(X, drop_rate=0.5):
    """
    Performs variational dropout on the rightmost dimension.
    :param X:
    :param drop_rate:
    :return:
    """
    rand_1d = tf.random_uniform([tf.shape(X)[-1],], dtype=np.float32)
    mask_1d = rand_1d < drop_rate

    rept = tf.reduce_prod(tf.shape(X))/tf.shape(X)[-1]
    boolean_mask = tf.reshape(tf.tile(mask_1d,[rept]), shape=tf.shape(X))

    return tf.where(boolean_mask, tf.zeros_like(X), X)


sess = tf.Session()
X_dropped = variational_dropout(X)
ret = sess.run(X_dropped)
print ret
