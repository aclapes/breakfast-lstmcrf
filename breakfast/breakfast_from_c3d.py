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

def generate_output(video_info, labels, length=16):
    ''' Given the info of the vide, generate a vector of classes corresponding
    the output for each clip of the video which features have been extracted.
    '''
    nb_frames = video_info['num_frames']
    last_first_name = nb_frames - length + 1

    start_frames = range(0, last_first_name, length // 2)

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

def create(features_path, stride, info_file, labels_file, output_dir):
    with open(info_file, 'r') as f:
        videos_data = json.load(f)
        videos_data = {
            key.replace(u'cereals', 'cereal').replace(u'salat', 'salad'): val for key, val in videos_data.iteritems()
        }
    with open(labels_file, 'r') as f:
        labels = import_labels(f)

    features_file = os.path.join(features_path, 'video_features.hdf5')
    f_video_features = h5py.File(features_file, 'r')

    dataset = dict()
    subsets = ['training', 'validation', 'testing']
    for subset in subsets:
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]

        videos = list(set(videos) & set(f_video_features.keys()))

        nb_videos = len(videos)
        print('Number of videos for {} subset: {}'.format(subset, nb_videos))

        dataset[subset] = dict(
            video_features = {},
            outputs = {},
            lengths = {}
        )
        for i,key in enumerate(videos):
            print('Reading %d/%d from disk...' % (i,nb_videos))
            dataset[subset]['video_features'][key] = np.array(f_video_features[key][...], dtype=np.float32)
            dataset[subset]['outputs'][key] = generate_output(videos_data[key], labels)
            dataset[subset]['lengths'][key] = len(dataset[subset]['outputs'][key])
            assert dataset[subset]['video_features'][key].shape[0] == dataset[subset]['lengths'][key]

    max_len = np.max([np.max(dataset[subset]['lengths'].values()) for subset in subsets])
    max_len = ((max_len // stride) + 1) * stride

    output_file = os.path.join(output_dir, 'dataset_stateful.hdf5')
    f_dataset = h5py.File(output_file, 'w')
    for subset in subsets:
        print('Creating HDF file for %s...' % (subset))
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]
        videos = list(set(videos) & set(f_video_features.keys()))


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

        f_dataset_subset = f_dataset.create_group(subset)
        f_dataset_subset.create_dataset(
            'video_features',
            data=video_features[perm,:,:],
            chunks=(16, video_features.shape[1], video_features.shape[2]),
            dtype='float32')
        f_dataset_subset.create_dataset(
            'outputs',
            data=outputs[perm,:],
            chunks=(16, outputs.shape[1]),
            dtype='float32')
        f_dataset_subset.create_dataset(
            'lengths',
            data=lengths[perm],
            chunks=(16,1),
            dtype='int32')

    # Save some additional attributes
    f_dataset.attrs['no_classes'] = len(labels)
    f_dataset.attrs['stride'] = stride

    f_dataset.close()

    # Sanity check
    f_dataset = h5py.File(output_file, 'r')
    f_dataset.attrs['no_classes'] == len(labels)
    f_dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breakfast hdf5 breakfast from feature files.')

    parser.add_argument(
        '-f',
        '--features-file',
        type=str,
        dest='features_file',
        default='/datasets/breakfast/3dconv_feats/video_features.h5',
        help=
        'File in which features are stored (default: %(default)s)')

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
        default='/datasets/breakfast/3dconv_feats/breakfast.h5',
        help=
        'Directory where hd5 file will be generated (default: %(default)s)')

    args = parser.parse_args()

    create(args.features_file,
           args.stride,
           args.videos_info,
           args.labels,
           args.output_file)