import tensorflow as tf
import numpy as np
import os

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

x_itgl = tf.cumsum(x_batch, axis=1, reverse=True)
x_itgl = tf.pad(x_itgl, [[0,0],[0,1],[1,0]], 'CONSTANT', constant_values=0.0)

p_batch = tf.constant(np.array([[0.8,0.45,0.70]]), dtype=tf.float64)

def sequence_score():
    betas_0 = tf.scatter_nd(
        tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0, 0], [0, 2]], 'CONSTANT', constant_values=0.0),
        tf.ones([batch_size], dtype=tf.float64) - tiniest_f64,
        shape=[batch_size, 1, num_classes],
    )
    betas_0 = tf.log(tf.cast(betas_0, dtype=tf.float64) + tiniest_f64)

    def should_continue(t, *args):
        return t <= time_steps

    def iteration(t, betas):
        t_norm = tf.range(tf.cast(t, tf.float64), 0, delta=-1, dtype=tf.float64)

        scores_ = (x_itgl[:, :t, 1:] - x_itgl[:, t:(t + 1), 1:])
        scores_norm = scores_ / tf.expand_dims(tf.expand_dims(t_norm, axis=-1), axis=0)

        tmp = scores_norm * tf.expand_dims(p_batch, 1)
        tmp_pt = tf.Print(tmp, [tmp], "Tmp is: ")
        # tmp2 = betas[:, :, :-1] + tmp
        tmp2 = betas[:,:,:-1] + tmp_pt
        tmp2_pt = tf.Print(tmp2, [tmp2], "Tmp2 is: ")

        betas_t = tf.reduce_logsumexp(tmp2_pt, axis=1, keep_dims=True)
        betas_t = tf.pad(betas_t, [[0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=np.log(tiniest_f64))

        return t + 1, tf.concat([betas, betas_t], axis=1)

    initial_t = tf.constant(1, dtype=tf.int32)

    t_final, betas = tf.while_loop(
        should_continue, iteration,
        loop_vars=[initial_t, betas_0],
        shape_invariants=[initial_t.get_shape(), tf.TensorShape([None, None, None])])

    return betas

betas = sequence_score()

def partition_function():
    alphas_0 = tf.scatter_nd(
        tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0, 0], [0, 2]], 'CONSTANT', constant_values=0.0),
        tf.ones([batch_size], dtype=tf.float64) - tiniest_f64,
        shape=[batch_size, 1, num_classes]
    )
    alphas_0 = tf.log(tf.cast(alphas_0, dtype=tf.float64) + tiniest_f64)

    transitions_ = tf.expand_dims(tf.expand_dims(transitions, axis=0), axis=0)

    def should_continue(t, *args):
        return t <= time_steps

    def iteration(t, alphas):
        # alphas_ = tf.expand_dims(alphas, 3)
        # alphas_print = tf.Print(alphas_, [t, tf.shape(alphas_)], 'tf.shape(tmp): ')

        # scores_ = tf.slice(x_itgl, [0,1,0], [-1,t,-1]) - tf.slice(x_itgl, [0,t+1,0], [-1,1,-1])
        t_norm = tf.range(tf.cast(t,tf.float64), 0, delta=-1, dtype=tf.float64)

        scores_ = (x_itgl[:, :t, :] - x_itgl[:, t:(t + 1), :])
        scores_norm = scores_ / tf.expand_dims(tf.expand_dims(t_norm, axis=-1), axis=0)

        scores_pt = tf.Print(scores_norm, [t, tf.shape(scores_), tf.shape(scores_norm), t_norm], "t shapes: ")

        # tmp = tf.expand_dims(scores_, axis=-1) * transition_params_
        # tmp = alphas_ + tf.cast(transition_params_, dtype=tf.float64)
        # tmp2 = tf.cast(scores_, dtype=tf.float64) + tf.reduce_logsumexp(tmp, axis=2)
        tmp = (tf.expand_dims(scores_pt, 2) * transitions_)  # [B,L,1,C] * [1,1,C,C]
        tmp2 = tmp + tf.expand_dims(alphas, 3)  # [B,L,C,C] * [B,L,C,1]

        # s0 * p00 +a0  s1 * p01 + a0
        # s0 * p10 +a1  s1 * p11 + a1
        # s0 * p20 +a2  s1 * p21 + a2

        tmp2 = tf.reduce_logsumexp(tmp2, axis=2)  # [B,L,C]

        alphas_t = tf.reduce_logsumexp(tmp2, axis=1, keep_dims=True)  # [B,1,C]
        alphas_t = tf.pad(alphas_t[:,:,1:], [[0,0],[0,0],[1,0]], 'CONSTANT', constant_values=np.log(tiniest_f64))
        return t + 1, tf.concat([alphas, alphas_t], axis=1)

    initial_t = tf.constant(1, dtype=tf.int32)

    t_final, alphas = tf.while_loop(
        should_continue, iteration,
        loop_vars=[initial_t, alphas_0],
        shape_invariants=[ initial_t.get_shape(), tf.TensorShape([None, None, None]) ]
    )

    return alphas

alphas = partition_function()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    betas_np, alphas_np = sess.run([betas, alphas])
    print 'Sequence scores: '
    print betas_np
    print 'Partition function:'
    print alphas_np