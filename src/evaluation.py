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

    return correct_labels / float(total_labels)

def compute_accuracy(predictions, labels):
    '''
    :param predictions: 1-D array of predictions [batch_size]
    :param labels: 1-D array of labels [batch_size]
    :return:
    '''

    correct_labels = np.sum(np.equal(predictions, labels))
    total_labels = len(predictions)

    return 100. * correct_labels / float(total_labels)

def compute_accuracy(predictions, labels):
    '''
    :param predictions: 1-D array of predictions [batch_size]
    :param labels: 1-D array of labels [batch_size]
    :return:
    '''

    correct_labels = np.sum(np.equal(predictions, labels))
    total_labels = len(predictions)

    return 100. * correct_labels / float(total_labels)


def compute_classwise_accuracy(predictions, labels, lengths, class_weights):
    '''
    Computes the framewise accuracy over a set of predictions.

    :param predictions: 2-D array of predictions [num_batches, num_timesteps]
    :param labels: 2-D array of labels [num_batches, num_timesteps]
    :param lengths: 1-D array of lengths [num_batches]. Used to trim padded timesteps.
    :return:
    '''

    hit_classes = np.zeros((len(class_weights),), dtype=np.float32)
    true_classes = np.zeros((len(class_weights),), dtype=np.float32)

    for pred, y, l in zip(predictions, labels, lengths):
        for c in range(len(class_weights)):
            hits = np.sum((pred[:l] == c) & (y[:l] == c))
            trues = np.sum(y[:l] == c)
            hit_classes[c] += hits
            true_classes[c] += trues

    return hit_classes, true_classes

def foo(predictions, labels, class_weights):
    '''
    Computes the framewise accuracy over a set of predictions.

    :param predictions: 2-D array of predictions [num_batches, num_timesteps]
    :param labels: 2-D array of labels [num_batches, num_timesteps]
    :param lengths: 1-D array of lengths [num_batches]. Used to trim padded timesteps.
    :return:
    '''

    hit_classes = np.zeros((len(class_weights),), dtype=np.float32)
    true_classes = np.zeros((len(class_weights),), dtype=np.float32)

    for c in range(len(class_weights)):
        hits = np.sum((predictions == c) & (labels == c))
        trues = np.sum(labels == c)
        hit_classes[c] += hits
        true_classes[c] += trues

    return hit_classes, true_classes