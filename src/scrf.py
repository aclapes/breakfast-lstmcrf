import tensorflow as tf
import numpy as np

x_ordered = None

LARGE_NEGATIVE = -99999

def scrf_log_likelihood(input, orderings, lengths, transition_matrix):
    """
    ...
    :param input: sequence scores (padded and shaped as [batch_size, max_seq_len, num_classes)
    :param orderings: orderings of actions (starting with a 0-labeled dummy class)
    :param lengths: length of the sequences
    :param transition_matrix: matrix transition probabilities between segments
    :param dtype: data type for computation (either tf.float32 or tf.float64)

    :return: loglikelihood, betas (sequence score DP matrix), and alphas (partition function DP matrix)
    """

    dtype = input.dtype

    # Integral sum over of logits over time (in reverse direction), so as to compute segment scores efficiently.
    input_igrl = tf.cumsum(input, axis=1, reverse=True)
    # Append some vectors of zeros:
    # * in dim=1: scores at time t can be input_igrl[:,t,:]-input_igrl[:,t+1,:]).
    # * in dim=2: to account for a dummy 0-th index class.
    input_igrl = tf.pad(input_igrl, [[0, 0], [0, 1], [1, 0]], 'CONSTANT', constant_values=tf.constant(0, dtype=dtype))

    # Do the magic
    sequences_scores, betas = scrf_sequence_score(input_igrl, orderings, lengths, transition_matrix)
    log_norm, alphas = scrf_log_norm(input_igrl, lengths, transition_matrix)

    return sequences_scores - log_norm, betas, alphas


def scrf_sequence_score(x_igrl, orderings, lengths, transition_matrix):
    '''
    Computes a sequence score (also referred as betas)
    :param x_igrl:
    :param orderings:
    :param lengths:
    :param transition_matrix:
    :return:
    '''
    batch_size  = tf.shape(x_igrl)[0]
    time_steps  = tf.shape(x_igrl)[1]

    x_dtype = x_igrl.dtype

    # Data preparation
    # ----------------
    # Each batch element corresponds to a sequence with a diff no. orderings. This requires some preparation
    # and data structures prior to batch computation.

    x_ordered_ta = tf.TensorArray(dtype=x_dtype, size=batch_size, name='x_ordered')  # x_igrl re-ordered using orderings
    transition_probs_ta = tf.TensorArray(dtype=x_dtype, size=batch_size, name='transition_probs')
    termination_inds_ta = tf.TensorArray(dtype=tf.int32, size=batch_size, name='termination_inds')

    # Define a tensorflow loop over batch elements (B)

    def cond(i, *args): return i < batch_size

    def loop(i, x_ordered_ta, transition_probs_ta, termination_inds_ta):
        # get segment labels within this sequence (batch element)
        orderings_valid = tf.boolean_mask(orderings[i], orderings[i] >= 0)  # -1 are padded order elements (mask them)
        nb_orderings = tf.shape(orderings_valid)[0]   # this sequence's number of orderings
        max_nb_orderings = tf.shape(orderings[i])[0]  # max number of orderings within this batch

        # Gather segment scores
        segm_scores = tf.gather(x_igrl[i,:,:], orderings_valid, axis=1) # this will construct a 2D tensor with T timesteps
                                                                        # and columns ordered given the orderings for this
                                                                        # sequence
        segm_scores_pad = tf.pad(segm_scores, [[0, 0], [0, max_nb_orderings-nb_orderings]],  # but later batch computation
                                 'CONSTANT', constant_values=tf.cast(0, dtype=x_dtype))      # requires all dimensions to
                                                                                             # to be equal size, so pad them.

        # Get the transition probabilities between contiguous segments
        # e.g. from orderings to transitions, e.g. "[0,1,34,3,..]" -> "[[0,1],[1,34],[34,3],[3,..
        transition_indices = tf.concat(
            [tf.expand_dims(orderings_valid[:-1], axis=1), tf.expand_dims(orderings_valid[1:], axis=1)],
            axis=1
        )
        trans_probs = tf.gather_nd(transition_matrix, transition_indices)  # look into transitions matrix
        trans_probs_pad = tf.pad(trans_probs, [[0, max_nb_orderings-nb_orderings]],
                                 'CONSTANT', constant_values=tf.cast(0, dtype=x_dtype))  # re-pad

        x_ordered_ta        = x_ordered_ta.write(i, segm_scores_pad)
        transition_probs_ta = transition_probs_ta.write(i, trans_probs_pad)
        termination_inds_ta = termination_inds_ta.write(i, tf.stack([i, lengths[i], nb_orderings])) # TODO: check if nb_orderings-1

        return i+1, x_ordered_ta, transition_probs_ta, termination_inds_ta

    _, x_ordered_ta, transition_probs_ta, termination_inds_ta = tf.while_loop(
        cond, loop,
        loop_vars=[0, x_ordered_ta, transition_probs_ta, termination_inds_ta])

    x_ordered = x_ordered_ta.stack()
    transitions_probs = transition_probs_ta.stack()
    termination_inds = termination_inds_ta.stack()

    # Calculate betas DP matrix
    # -------------------------
    # This processes all batch elements at once.

    # Initialize betas (quite tricky).
    # Since operations are performed in logspace (also logsumexp used), it basically initializes all betas to
    # LARGE_NEGATIVE except for the [i,0,0] forall i.
    betas_0 = tf.scatter_nd(
        tf.pad(tf.expand_dims(tf.range(batch_size), axis=1), [[0, 0], [0, 2]], 'CONSTANT', constant_values=0),
        tf.zeros([batch_size], dtype=x_dtype) - LARGE_NEGATIVE,
        shape=[batch_size, 1, tf.shape(x_ordered)[-1]],
    ) + LARGE_NEGATIVE

    # Define a while loop over the temporal dimension (T)

    def cond(t, *args): return t < time_steps

    def loop(t, betas):
        # Slice scores and 'undo' the reverse integral sum up to time t" by sutracting the scores at "t+1".
        scores = x_ordered[:,:t,:] - x_ordered[:,t:(t+1),:]

        # A segment can be represented as the sum of frame scores in it or the mean of scores.
        # Keep the following two lines commented to use the sum.
        # ---
        # t_norm = tf.cast(tf.range(t, 0, delta=-1), dtype=x_dtype)
        # scores = scores / tf.expand_dims(tf.expand_dims(t_norm, axis=-1), axis=0)
        # ---

        tmp = scores[:,:,1:] + tf.expand_dims(transitions_probs, 1)  # [B,L,S-1] .+ [B,1,S-1]. It omits dummy class scores.
        tmp2 = betas[:,:,:-1] + tmp

        betas_t = tf.reduce_logsumexp(tmp2, axis=1, keep_dims=True)
        betas_t = tf.pad(betas_t, [[0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=LARGE_NEGATIVE)

        return t+1, tf.concat([betas, betas_t], axis=1)

    _, betas_f = tf.while_loop(
        cond, loop,
        loop_vars=[1, betas_0],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None, None])])

    betas = tf.gather_nd(betas_f, termination_inds)  # termination indexes the last timestep of sequence and last class.
                                                     # (sequences are paddes, so we needed to keep track of this info)

    return betas, betas_f  # betas_f is returned for debugging purposes


def scrf_log_norm(x_igrl, lengths, transition_matrix):
    batch_size  = tf.shape(x_igrl)[0]  # B
    time_steps  = tf.shape(x_igrl)[1]  # T
    num_classes = tf.shape(x_igrl)[2]  # C

    x_dtype = x_igrl.dtype

    alphas_0 = tf.scatter_nd(
        tf.pad(tf.expand_dims(tf.range(batch_size), axis=-1), [[0, 0], [0, 2]],
               'CONSTANT', constant_values=0.0),
        tf.zeros([batch_size], dtype=x_dtype) - LARGE_NEGATIVE,
        shape=[batch_size, 1, num_classes]
    ) + LARGE_NEGATIVE

    transitions_matrix_ = tf.expand_dims(tf.expand_dims(transition_matrix, axis=0), axis=0)
    # Create a mask we'll use to avoid auto-transitions.
    # (we cannot move from a segment with k class to the same k class)
    # mask = tf.expand_dims(tf.expand_dims(tf.eye(num_classes, num_classes, dtype=x_dtype) * LARGE_NEGATIVE, axis=0), axis=0)

    def should_continue(t, *args):
        return t < time_steps

    def iteration(t, alphas):
        scores = x_igrl[:,:t,:] - x_igrl[:,t:(t+1),:]

        # t_norm = tf.range(tf.cast(t,dtype=x_dtype), 0, delta=-1, dtype=x_dtype)
        # scores = scores / tf.expand_dims(tf.expand_dims(t_norm, axis=-1), axis=0)

        # Perform the following computation:
        # s_0 * p_00 + a_0   s_1 * p_01 + a_0
        # s_0 * p_10 + a_1   s_1 * p_11 + a_1
        # s_0 * p_20 + a_2   s_1 * p_21 + a_2
        # where s_k = s_{t,k} is score of k-th class at time t (omitted), p_ij is prob of transitioning from i->k,
        # a_{t-1,k} = a_k is alpha value on class k a previous time step t-1 (omitted).

        tmp = tf.expand_dims(scores, 2) + transitions_matrix_  # [B,L,1,C] + [1,1,C,C]
        tmp2 = tmp + tf.expand_dims(alphas, 3)  # [B,L,C,C] * [B,L,C,1]

        tmp2 = tf.reduce_logsumexp(tmp2, axis=2)  # [B,L,C]

        alphas_t = tf.reduce_logsumexp(tmp2, axis=1, keep_dims=True)  # [B,1,C]
        alphas_t = tf.pad(alphas_t[:,:,1:], [[0,0],[0,0],[1,0]], 'CONSTANT', constant_values=LARGE_NEGATIVE)
        alphas_t = (alphas_t)

        return t+1, tf.concat([alphas, alphas_t], axis=1)

    t_final, alphas_final = tf.while_loop(
        should_continue, iteration,
        loop_vars=[1, alphas_0],
        shape_invariants=[ tf.TensorShape([]), tf.TensorShape([None, None, None]) ]
    )

    # Get alpha_T of each sequence (recall they are padded to max sequence length in batch)
    # <--
    mask = tf.one_hot(lengths, time_steps, on_value=True, off_value=False, dtype=tf.bool)
    alphas_T = tf.boolean_mask(alphas_final, mask)
    # --- (funkier older code)
    # alphas = tf.gather(alphas_final, lengths, axis=1)
    # alphas_T = tf.boolean_mask(alphas, tf.cast(tf.eye(tf.shape(alphas)[1]), dtype=tf.bool))
    # -->

    alphas = tf.reduce_logsumexp(alphas_T, axis=1)

    return alphas, alphas_final


def scrf_viterbi_dp(x_igrl, lengths, transition_matrix):
    x_igrl = tf.cumsum(x_igrl, axis=1, reverse=True)
    x_igrl = tf.pad(x_igrl, [[0, 0], [0, 1], [1, 0]], 'CONSTANT', constant_values=tf.constant(0, dtype=x_igrl.dtype))
    transitions_matrix_ = tf.expand_dims(tf.expand_dims(transition_matrix, axis=0), axis=0)

    batch_size  = tf.shape(x_igrl)[0]
    time_steps  = tf.shape(x_igrl)[1]
    num_classes = tf.shape(x_igrl)[2]

    x_dtype = x_igrl.dtype

    vis_0 = tf.zeros([batch_size, 1, num_classes], dtype=x_dtype)
    backpointers_0 = tf.zeros([batch_size, 1, num_classes, 2], dtype=tf.int32)

    def should_continue(t, *args): return t < time_steps

    def iteration(t, vis, bps):
        scores = (x_igrl[:,:t,:] - x_igrl[:,t:(t+1),:])

        # t_norm = tf.range(tf.cast(t,dtype=x_dtype), 0, delta=-1, dtype=x_dtype)
        # scores = scores / tf.expand_dims(tf.expand_dims(t_norm, axis=-1), axis=0)

        tmp = tf.expand_dims(scores, 2) + transitions_matrix_  # [B,L,1,C] * [1,1,C,C]
        tmp2 = tmp + tf.expand_dims(vis, 3)  # [B,L,C,C] * [B,L,C,1]

        # Reshape prior to apply max and argmax operations
        tmp3 = tf.reshape(tmp2,[tf.shape(tmp2)[0],-1,tf.shape(tmp2)[3]])

        # Viterbi values (max)
        vis_t = tf.reduce_max(tmp3, axis=1, keep_dims=True)
        vis_t = tf.pad(vis_t[:,:,1:], [[0,0],[0,0],[1,0]], 'CONSTANT')

        # Backpointers (argmax)
        argmaxes = tf.cast(tf.argmax(tmp3, axis=1), tf.int32)
        bps_t = tf.stack(
            [argmaxes // tf.shape(tmp2)[2], tf.mod(argmaxes, tf.shape(tmp2)[2])]
            , axis=-1
        )
        bps_t = tf.expand_dims(bps_t, axis=1)

        return t+1, tf.concat([vis, vis_t], axis=1), tf.concat([bps, bps_t], axis=1)

    _, vis_f, backpointers_f = tf.while_loop(
        should_continue, iteration,
        loop_vars=[1, vis_0, backpointers_0],
        shape_invariants=[ tf.TensorShape([]),
                           tf.TensorShape([None, None, None]),
                           tf.TensorShape([None, None, None, 2]) ]
    )

    # This returns a viterbi DP matrix and backpointers. Decoding is done with a python function, see decode_viterbi_dp).

    return vis_f, backpointers_f


def decode_viterbi_dp(viterbi, backpointers, lengths):
    """
    Non-tensorflow function to be used on the output of scrf_viterbi_dp function
    :param viterbi:
    :param backpointers:
    :param lengths:
    :return:
    """
    assert viterbi.shape == backpointers.shape[:-1]
    maxlen = viterbi.shape[1] - 1  # "-1" for the dummy initial timestep transitioning from 0-th to any

    decodings = []  # decoding paths
    for i,(V,B) in enumerate(zip(viterbi, backpointers)):
        curr_time = lengths[i]
        curr_lbl = np.argmax(V[curr_time,:])
        decoding_i = []
        while curr_time > 0:
            next_time, next_lbl = B[curr_time, curr_lbl]
            decoding_i += [curr_lbl] * (curr_time - next_time)
            curr_time, curr_lbl = next_time, next_lbl
        decoding_i.reverse()  # recall we are going backwards
        decoding_i += [-1] * (maxlen - len(decoding_i)) # pad to same length with minus ones (-1s)
        decodings.append(decoding_i)

    return np.array(decodings)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    # Generate toy data to test the method

    # Sequence scores for actions
    # [2 batch x 3 classes x 4 timesteps] . 1 dummy zero-labeled class must be added.
    x_batch = tf.constant(np.array(
        [
            [[0.55, 0.40, 0.05],
             [0.21, 0.59, 0.10],
             [0.20, 0.20, 0.00],
             [0.98, 0.01, 0.01],
             [0.00, 0.00, 0.00]] ,

            [[0.05, 0.95, 0.00],
             [0.21, 0.59, 0.10],
             [0.20, 0.20, 0.00],
             [0.98, 0.01, 0.01],
             [0.07, 0.91, 0.02]],
        ]), dtype=tf.float64)

    # Suppose the action orderings in this0->1->2->3
    # [2 batch x 4 timesteps]
    orderings = tf.constant(np.array([[0,1,2,3],[0,2,1,-1]]))

    # [2 batch]
    lengths = tf.constant([4,5])

    # Transition probs among (1+3) classes.
    # [(1+3) classes x (1+3) classes]
    # Position (i,j) is the probability of t=j and t-1=i
    transitions = tf.constant(np.array([[0.00, 0.80, 0.10, 0.10],
                                        [0.00, 0.00, 0.45, 0.55],
                                        [0.00, 0.30, 0.00, 0.70],
                                        [0.00, 0.60, 0.40, 0.00]]), dtype=tf.float64)

    loglikelihood_op, betas, alphas = scrf_log_likelihood(x_batch, orderings, lengths, transitions)
    cost_op = -tf.reduce_mean(loglikelihood_op, axis=0)

    viterbi_op = scrf_viterbi_dp(x_batch, lengths, transitions)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        logLL, betas, alphas = sess.run([cost_op, betas, alphas])
        print logLL
        print alphas
        print betas
        viterbi, backpointers = sess.run(viterbi_op)
        alignment = decode_viterbi_dp(viterbi, backpointers, [4, 5])
        print alignment
