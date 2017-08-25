import numpy as np

def read_data_generator(data, labels, lengths, batch_size=16):
    '''
    This generator function serves a batch of the dataset at each call.
    See what a generator function is ;)
    :param data:
    :param labels:
    :param lengths:
    :param batch_size:
    :return:
    '''

    n_batches = len(data) // batch_size  # this will discard the last batch

    for i in range(n_batches):
        # prepare the batch
        x = data[(i*batch_size):((i+1)*batch_size),:,:] # batch features
        y = labels[(i * batch_size):((i + 1) * batch_size), :] # batch labels
        l = lengths[(i * batch_size):((i + 1) * batch_size), :]  # not returned!

        yield (x, y, np.squeeze(l))


# def read_data_generator(data, labels, lengths, batch_size=16):
#     '''
#     This generator function serves a batch of the dataset at each call.
#     See what a generator function is ;)
#     :param data:
#     :param labels:
#     :param lengths:
#     :param batch_size:
#     :return:
#     '''
#
#     n_batches = len(data) // batch_size  # this will discard the last batch
#
#     for i in range(n_batches):
#         # prepare the batch
#         x = data[(i*batch_size):((i+1)*batch_size),:,:] # batch features
#         y = labels[(i * batch_size):((i + 1) * batch_size), :] # batch labels
#
#         # instead of using lengths, create a binary mask (to mask padded timesteps)
#         l = lengths[(i * batch_size):((i + 1) * batch_size)]  # not returned!
#
#         # batch mask preparation
#         w = np.zeros((l.shape[0], y.shape[1]), dtype=np.float32)
#         for k in range(l.shape[0]):
#             l_k = int(l[k])  # length of the k-th seq in the batch
#             w[k, :l_k] = 1.  # binary mask for k-th seq
#
#         yield (x, y, w)