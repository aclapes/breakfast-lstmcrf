import tensorflow as tf
import numpy as np
import os

# from src import scrf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tiniest_f64 = np.finfo(np.float64).tiny
tiniest_f32 = np.finfo(np.float32).tiny

np.set_printoptions(precision=4)

# Generate some data
# ==================

# Per-timestep scores
x_batch = tf.constant(np.array([[[0.55, 0.40, 0.05],
                                 [0.21, 0.59, 0.10],
                                 [0.20, 0.20, 0.00],
                                 [0.98, 0.01, 0.01]]]), dtype=tf.float64)
# batch_size = tf.shape(x_batch)[0]
# time_steps = tf.shape(x_batch)[1]
# num_classes = 1 + tf.shape(x_batch)[2]
batch_size = 1
num_classes = 4
time_steps = 4

transitions = tf.constant(np.array([[0.00, 0.80, 0.10, 0.10],
                                    [0.00, 0.00, 0.45, 0.55],
                                    [0.00, 0.30, 0.00, 0.70],
                                    [0.00, 0.60, 0.40, 0.00]]), dtype=tf.float64)

# Generate some data
# ==================



p_batch = tf.constant(np.array([[0.8,0.45,0.70]]), dtype=tf.float64)



if __name__ == '__main__':

    # betas = scrf.scrf_sequence_score()
    # alphas = scrf.scrf_log_norm()

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print sess.run(tf.constant(0, tf.float64))
        # betas_np, alphas_np = sess.run([betas, alphas])
        # print 'Sequence scores: '
        # print betas_np
        # print 'Partition function:'
        # print alphas_np