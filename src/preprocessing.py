import tensorflow as tf
from reader import read_labels_generator, read_image_generator
from progressbar import ProgressBar
import numpy as np

def compute_class_weights(input_data, no_classes, batch_size):
    graph = tf.Graph()
    with graph.as_default():
        y_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        y = tf.one_hot(y_placeholder, no_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_sum = tf.reduce_sum(y, axis=[0, 1])

    with tf.Session(graph=graph) as session:
        reader = read_labels_generator(input_data['outputs'],
                                       batch_size=batch_size)

        num_instances = input_data['outputs'].shape[0]
        num_batches = int(np.ceil(num_instances / float(batch_size)))

        class_counts_acc = np.zeros((no_classes,), dtype=np.float64)
        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = reader.next()
            ret = session.run([y_sum], feed_dict={y_placeholder: batch})
            class_counts_acc += ret[0]
            progbar.update(b)
        progbar.finish()

        return np.max(class_counts_acc) / class_counts_acc

def compute_mean_channels(input_data, batch_size):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, None, None, None])
        channel_means_op = tf.reduce_mean(tf.cast(x, tf.float32), axis=[0, 1, 2])

    with tf.Session(graph=graph) as session:
        reader = read_image_generator(input_data['video_features'],
                                      input_data['outputs'],
                                      batch_size=batch_size)

        num_instances = input_data['video_features'].shape[0]
        num_batches = int(np.ceil(num_instances / float(batch_size)))

        channel_means_acc = np.zeros((3,), dtype=np.float64)
        progbar = ProgressBar(max_value=num_batches)
        for b in range(num_batches):
            batch = reader.next()
            ret = session.run([channel_means_op], feed_dict={x: batch[0]})
            channel_means_acc += ret[0]
            progbar.update(b)
        progbar.finish()

        return channel_means_acc / num_batches