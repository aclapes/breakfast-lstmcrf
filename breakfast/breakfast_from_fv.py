import h5py
import argparse
import json
import os
import numpy as np


def import_labels(f):
    ''' Read from a file all the labels from it '''
    lines = f.readlines()
    labels = []
    i = 0
    for l in lines:
        t = l.split('\t')
        assert int(t[0]) == i
        label = t[1].split('\n')[0]
        labels.append(label)
        i += 1
    return labels

def read_features(filepath, pool_op, length=8):
    import csv
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        X = np.array([l for l in reader], dtype=np.float32)[:,1:]
    X = X[:(X.shape[0]//length)*length,:]

    if pool_op == 'avg':  # average pooling
        X = np.mean(X.reshape(X.shape[0]//length, length, X.shape[1]), axis=1)
    elif pool_op == 'max':  # max pooling
        X_r = X.reshape(X.shape[0]//length, length, X.shape[1])
        X_min, X_max = X_r.min(axis=1)
        X = np.where(-X_min > X_max, X_min, X_max)

    return X

def generate_output(video_info, labels, length=16):
    ''' Given the info of the vide, generate a vector of classes corresponding
    the output for each clip of the video which features have been extracted.
    '''
    nb_frames = video_info['num_frames']
    last_first_name = nb_frames - length + 1

    start_frames = range(0, last_first_name, length)

    # Check the output for each frame of the video
    outputs = ['none'] * nb_frames
    for i in range(nb_frames):
        # Pass frame to temporal scale
        t = i / float(nb_frames) * video_info['duration']
        for annotation in video_info['annotations']:
            if t >= annotation['segment'][0] and t <= annotation['segment'][1]:
                outputs[i] = annotation['label']
                label = annotation['label']
                break

    instances = []
    for start_frame in start_frames:
        outs = outputs[start_frame:start_frame + length]
        _, counts = np.unique(outs, return_counts=True)
        label = outs[np.argmax(counts)]

        output = labels.index(label)

        instances.append(output)

    return instances

def create(features_path, pool_op, pool_size, stride, info_file, labels_file, output_dir):
    with open(info_file, 'r') as f:
        videos_data = json.load(f)
        # uncomment if want to merge validation and testing
        # ---
        # for key, value in videos_data.iteritems():
        #     if videos_data[key]['subset'] == 'validation':
        #         videos_data[key]['subset'] = 'training'
        # ---

    with open(labels_file, 'r') as f:
        labels = import_labels(f)

    dataset = dict()
    subsets = ['training', 'testing', 'validation']

    class_counts = np.zeros((len(labels),), dtype=np.float32)

    for subset in subsets:
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]

        nb_videos = len(videos)
        print('Number of videos for {} subset: {}'.format(subset, nb_videos))

        dataset[subset] = dict(
            video_features = {},
            outputs = {},
            lengths = {}
        )

        for i,key in enumerate(videos):
            print('Reading %d/%d from disk...' % (i,nb_videos-1))
            filepath = os.path.join(features_path + key.split('_')[-1], key + '.txt')
            x = read_features(filepath, pool_op, length=pool_size)
            y = generate_output(videos_data[key], labels, length=pool_size)

            if subset == 'training':  # count to assign class weights later
                ids, counts = np.unique(y, return_counts=True)
                for id,c in zip(ids,counts): class_counts[id] += c

            dataset[subset]['video_features'][key] = np.concatenate([x, np.zeros((len(y)-x.shape[0],x.shape[1]))])
            dataset[subset]['outputs'][key] = y
            dataset[subset]['lengths'][key] = len(dataset[subset]['outputs'][key])
            assert dataset[subset]['video_features'][key].shape[0] == len(dataset[subset]['outputs'][key])

    max_len = np.max([np.max(dataset[subset]['lengths'].values()) for subset in subsets])
    max_len = ((max_len // stride) + 1) * stride

    for subset in subsets:
        output_file = os.path.join(output_dir, subset + '.h5')
        f_dataset = h5py.File(output_file, 'w')

        print('Creating HDF file for %s...' % (subset))
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]

        num_features = dataset[subset]['video_features'][videos[0]].shape[1]

        video_features = np.zeros((len(videos), max_len, num_features), dtype=np.float32)
        outputs = np.zeros((len(videos), max_len), dtype=np.float32)
        lengths = np.zeros((len(videos), 1), dtype=np.int32)
        for i, key in enumerate(videos):
            leng = dataset[subset]['lengths'][key]
            video_features[i,:leng,:] = dataset[subset]['video_features'][key]
            outputs[i,:leng] = dataset[subset]['outputs'][key]
            lengths[i,0] = leng

        perm = np.random.RandomState(42).permutation(len(videos))

        f_dataset.create_dataset(
            'video_features',
            data=video_features[perm,:,:],
            chunks=(10, video_features.shape[1], video_features.shape[2]),
            dtype='float32')
        f_dataset.create_dataset(
            'outputs',
            data=outputs[perm,:],
            chunks=(10, outputs.shape[1]),
            dtype='float32')
        f_dataset.create_dataset(
            'lengths',
            data=lengths[perm],
            chunks=(10,1),
            dtype='int32')

        f_dataset.create_dataset('class_weights', data=(np.max(class_counts)/class_counts))

        # Save some additional attributes
        f_dataset.attrs['no_classes'] = len(labels)
        f_dataset.attrs['pool_op'] = pool_op
        f_dataset.attrs['pool_size'] = pool_size
        f_dataset.attrs['stride'] = stride
        # f_dataset.attrs['class_weights'] = (np.max(class_counts)/class_counts)

        f_dataset.close()

        # Sanity check
        f_dataset = h5py.File(output_file, 'r')
        # f_dataset.attrs['class_weights']
        assert f_dataset.attrs['no_classes'] == len(labels)
        f_dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breakfast hdf5 breakfast from feature files.')

    parser.add_argument(
        '-d',
        '--features-dir',
        type=str,
        dest='features_dir',
        default='/data/datasets/breakfast/fv/s1/',
        help=
        'Directory where features are stored (default: %(default)s)')

    parser.add_argument(
        '-i',
        '--videos-info',
        type=str,
        dest='videos_info',
        default='breakfast/videos.json',
        help=
        'File (json) where info of the videos is stored (default: %(default)s)')

    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        dest='labels',
        default='breakfast/labels.txt',
        help=
        'File (txt) where labels are listed (default: %(default)s)')

    parser.add_argument(
        '-po',
        '--pool-op',
        type=str,
        dest='pool_op',
        default='avg',
        help=
        'Pooling operation (avg or max) (default: %(default)s)')

    parser.add_argument(
        '-ps',
        '--pool-size',
        type=int,
        dest='pool_size',
        default=1,
        help=
        'Pooling stride (default: %(default)s)')

    parser.add_argument(
        '-s',
        '--stride',
        type=int,
        dest='stride',
        default=20,
        help=
        'Pad the end of the sequence, so no. steps is multiple of s (default: %(default)s)')

    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        dest='output_file',
        default='breakfast/dataset/',
        help=
        'Directory where hd5 file will be generated (default: %(default)s)')

    args = parser.parse_args()
    print args

    create(args.features_dir,
           args.pool_op,
           args.pool_size,
           args.stride,
           args.videos_info,
           args.labels,
           args.output_file)