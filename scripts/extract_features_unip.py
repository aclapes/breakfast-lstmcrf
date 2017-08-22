from __future__ import absolute_import

import argparse
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np
from progressbar import ProgressBar

import h5py
from src.data_unip import VideoGenerator


def extract_features(videos_dir, video_format, output_dir, batch_size, num_threads,
                     queue_size, num_gpus):
    # Defining variables
    input_size = (112, 112)
    length = 16

    output_path = os.path.join(output_dir, 'video_features.hdf5')
    mode = 'r+' if os.path.exists(output_path) else 'w'
    # Extract the ids of the videos already extracted its features
    output_file = h5py.File(output_path, mode)
    extracted_videos = list(output_file.keys())
    output_file.close()

    videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[-4:] == '.{0}'.format(video_format)]

    # Lets remove from the list videos_ids, the ones already extracted its features
    videos_ids_to_extract = list(set(videos_ids) - set(extracted_videos))

    nb_videos = len(videos_ids_to_extract)
    print('Total number of videos: {}'.format(len(videos_ids)))
    print('Videos already extracted its features: {}'.format(
        len(extracted_videos)))
    print('Videos to extract its features: {}'.format(nb_videos))

    # Creating Parallel Fetching Video Data

    generator = VideoGenerator(videos_ids_to_extract, videos_dir, video_format, length, input_size)

    # Loading the model
    print('Loading model')
    model = C3D_conv_features(summary=True)
    print('Compiling model')
    model.compile(optimizer='sgd', loss='mse')
    print('Compiling done!')

    print('Starting extracting features')

    print('Loading mean')
    mean_total = np.load('data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    has_stopped = False
    while not has_stopped:
        try:
            video_id, X = next(generator)

            st_time = time.time()
            X = X - mean
            features = model.predict(X, batch_size=batch_size)
            end_time = time.time()
            print('Extracted features from video {}: {:.2f} seconds'.format(video_id, end_time-st_time))

            with h5py.File(output_path, 'r+') as f:
                f.create_dataset(video_id, data=features, dtype='float32')
            print('Saved video {}'.format(video_id))

        except StopIteration:
            has_stopped = True



def C3D_conv_features(summary=False):
    """ Return the Keras model of the network until the fc6 layer where the
    convolutional features can be extracted.
    """
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential

    model = Sequential()
    # 1st layer group
    model.add(
        Convolution3D(
            64,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv1',
            subsample=(1, 1, 1),
            input_shape=(3, 16, 112, 112),
            trainable=False))
    model.add(
        MaxPooling3D(
            pool_size=(1, 2, 2),
            strides=(1, 2, 2),
            border_mode='valid',
            name='pool1'))
    # 2nd layer group
    model.add(
        Convolution3D(
            128,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv2',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool2'))
    # 3rd layer group
    model.add(
        Convolution3D(
            256,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv3a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        Convolution3D(
            256,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv3b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool3'))
    # 4th layer group
    model.add(
        Convolution3D(
            512,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv4a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        Convolution3D(
            512,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv4b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool4'))
    # 5th layer group
    model.add(
        Convolution3D(
            512,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv5a',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(
        Convolution3D(
            512,
            3,
            3,
            3,
            activation='relu',
            border_mode='same',
            name='conv5b',
            subsample=(1, 1, 1),
            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(
        MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            border_mode='valid',
            name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    model.load_weights('data/models/c3d-sports1M_weights.h5')

    for _ in range(4):
        model.pop_layer()

    if summary:
        print(model.summary())
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract video features using C3D network')
    parser.add_argument(
        '-d',
        '--videos-dir',
        type=str,
        dest='directory',
        default='data/videos',
        help='videos directory (default: %(default)s)')
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        dest='format',
        default='mp4',
        help=
        'set the video format'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        dest='output',
        default='data/dataset',
        help=
        'directory where to store the extracted features (default: %(default)s)'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=48,
        help='batch size when extracting features (default: %(default)s)')
    parser.add_argument(
        '-t',
        '--num-threads',
        type=int,
        dest='num_threads',
        default=8,
        help='number of threads to fetch videos (default: %(default)s)')
    parser.add_argument(
        '-q',
        '--queue-size',
        type=int,
        dest='queue_size',
        default=12,
        help=
        'maximum number of elements at the queue when fetching videos (default %(default)s)'
    )
    parser.add_argument(
        '-g',
        '--num-gpus',
        type=int,
        dest='num_gpus',
        default=1,
        help=
        'number of gpus to use for extracting features (default: %(default)s)')

    args = parser.parse_args()

    extract_features(args.directory, args.format, args.output, args.batch_size,
                     args.num_threads, args.queue_size, args.num_gpus)
