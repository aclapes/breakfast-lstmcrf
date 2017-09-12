import tensorflow as tf

def variational_dropout(X, drop_rate=0.5):
    """
    Performs variational dropout on the rightmost dimension.
    :param X:
    :param drop_rate:
    :return:
    """
    rand_1d = tf.random_uniform([tf.shape(X)[-1],], dtype=tf.float32)
    mask_1d = rand_1d < drop_rate

    rept = tf.reduce_prod(tf.shape(X))/tf.shape(X)[-1]
    boolean_mask = tf.reshape(tf.tile(mask_1d,[rept]), shape=tf.shape(X))

    return tf.where(boolean_mask, tf.zeros_like(X), X)