import numpy as np

def compute_framewise_accuracy(predictions, labels, lengths):
    '''
    Computes the framewise accuracy over a set of predictions.

    :param predictions: 2-D array of predictions [num_batches, num_timesteps]
    :param labels: 2-D array of labels [num_batches, num_timesteps]
    :param lengths: 1-D array of lengths [num_batches]. Used to trim padded timesteps.
    :return:
    '''

    correct_labels = total_labels = 0.

    for pred, y, l in zip(predictions, labels, lengths):
        correct_labels += np.sum(np.equal(pred[:l], y[:l]))
        total_labels += l

    return 100. * correct_labels / float(total_labels)


# def compute_framewise_accuracy(predictions, labels, weights):
#     '''
#     Computes the framewise accuracy over a set of predictions.
#
#     :param predictions: 2-D array of predictions [num_batches, num_timesteps]
#     :param labels: 2-D array of labels [num_batches, num_timesteps]
#     :param weights: 2-D array of weights [num_batches, num_timesteps]. Used to mask padded timesteps.
#     :return:
#     '''
#
#     correct_labels = total_labels = 0.
#
#     for pred, y, w in zip(predictions, labels, weights):
#         length = int(np.sum(w))
#         correct_labels += np.sum(np.equal(pred[:length], y[:length]))
#         total_labels += length
#
#     return 100. * correct_labels / float(total_labels)
