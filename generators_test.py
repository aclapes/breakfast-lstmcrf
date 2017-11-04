import tensorflow as tf
import numpy as np
import os
import h5py

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#
# x = tf.random_normal([100*2])
# y = tf.random_normal([100*2])
# x_good = tf.boolean_mask(x, tf.logical_and(tf.greater_equal(x, -1), tf.less_equal(x, 1)))
# y_good = tf.boolean_mask(y, tf.logical_and(tf.greater_equal(y, -1), tf.less_equal(y, 1)))
# p = tf.concat([tf.expand_dims(tf.slice(x_good, [0], [100]), axis=-1),
#                tf.expand_dims(tf.slice(y_good, [0], [100]), axis=-1)],
#               axis=1)
#
# # init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     pp = sess.run(p)
#     print pp
#     print pp.shape

f = h5py.File('test.test', 'w')
dt = h5py.special_dtype(vlen=np.dtype('int32'))
dset = f.create_dataset('vlen_int', (100,), dtype=dt)
dset[0] = [[1,2,3],[1,2,4]]
dset[1] = [1,2,3,4,5]
print dset[0]
f.close()