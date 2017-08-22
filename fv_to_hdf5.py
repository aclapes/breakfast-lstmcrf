import h5py
import argparse
import json
import os
import numpy as np

step_size = 20

def read_features(filepath, length=8):
    import csv
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        X = np.array([l for l in reader], dtype=np.float32)
    X = X[:(X.shape[0]//length)*length,:]

    X = np.mean(X.reshape(X.shape[0]//length, length, X.shape[1]), axis=1)

    return X

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

def generate_output(video_info, labels, length=8):
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

def create(path_features, info_file, labels_file, output_dir):
    with open(info_file, 'r') as f:
        videos_data = json.load(f)
    with open(labels_file, 'r') as f:
        labels = import_labels(f)

    dataset = dict()
    subsets = ['training', 'validation', 'testing']
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
            filepath = os.path.join(path_features + key.split('_')[-1], key + '.txt')
            x = read_features(filepath)
            y = generate_output(videos_data[key], labels)
            dataset[subset]['video_features'][key] = np.concatenate([x, np.zeros((len(y)-x.shape[0],x.shape[1]))])
            dataset[subset]['outputs'][key] = y
            dataset[subset]['lengths'][key] = len(dataset[subset]['outputs'][key])
            assert dataset[subset]['video_features'][key].shape[0] == len(dataset[subset]['outputs'][key])

    max_len = np.max([np.max(dataset[subset]['lengths'].values()) for subset in subsets])
    max_len = ((max_len // step_size) + 1) * step_size

    output_file = os.path.join(output_dir, 'dataset.hdf5')
    f_dataset = h5py.File(output_file, 'w')
    for subset in subsets:
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
    f_dataset.close()

    # pattern = re.compile(pattern_select)
    #
    # for action_dir in os.listdir(path_features):
    #     count = 0
    #     txt_files = os.listdir(os.path.join(path_features, action_dir))
    #
    #     for feats_filename in txt_files:
    #
    #         res = pattern.match(feats_filename)
    #         if res is None:
    #             print('Skipping file: %s' % feats_filename)
    #             continue
    #
    #         if count/float(len(txt_files)) > p: break  # DEBUG (load only the first N files)
    #         count += 1
    #
    #         feat_filepath = os.path.join(path_features, action_dir, feats_filename)
    #         segm_filepath = os.path.join(path_seg, action_dir, os.path.splitext(feats_filename)[0]+'.xml')
    #         filepaths.append((feat_filepath,segm_filepath))
    #
    # if shuffle:
    #     np.random.shuffle(filepaths)
    #
    # features = []
    # labels = []
    # transitions = []
    # subwords = dict()
    # sentences = []
    #
    # for feat_filepath, segm_filepath in filepaths:
    #     # Load the annotations in two forms: transitions and sentences of words
    #     transitions.append([])
    #     # sentences.append([])
    #
    #     with open(segm_filepath, 'r') as xmlfile:
    #         xml_lines = xmlfile.readlines()[2:-2]
    #         for line in xml_lines:
    #             name, st, end = re.findall('"([^"]*)"', line)
    #             st, end = int(st), int(end)
    #             transitions[-1].append((name, st, end))
    #             sentences += ([name] * (end-st+1))
    #             # if name not in subwords[action_dir]: subwords[action_dir][name] = 0
    #             # else: subwords[action_dir][name] += 1
    #             subwords.setdefault(name, 0)
    #             subwords[name] += 1
    #
    #     # Load the actual features
    #     print('Loading file: %s' % feat_filepath),
    #     with open(feat_filepath, 'r') as csvfile:
    #         reader = csv.reader(csvfile, delimiter='\t')
    #         X = np.array([l for l in reader], dtype=np.float32)[:,1:]
    #         X[np.isnan(X)] = 0.  # needed?
    #
    #     # Fix sequence and annotations length missmatch
    #     lastIdx = transitions[-1][-1][2]
    #     if lastIdx - X.shape[0] > 0:
    #         # add dummy (zero) lines
    #         print('-> PADDED')
    #         X = np.concatenate([X, np.zeros((lastIdx-X.shape[0],X.shape[1]), dtype=X.dtype)])
    #     else:
    #         print('-> TRUNCATED')
    #         # truncate the sequence to the last annotated frame
    #         X = X[:lastIdx,:]
    #
    #     features += [X[max(0,i+1-20):i+1,:] for i in range(0,X.shape[0])]
    #
    #     # if normalization == 'std':
    #     #     scaler = preprocessing.StandardScaler().fit(X[np.sum(X, 1) != 0., :])
    #     #     X = scaler.transform(X)
    #     # elif normalization == 'l1':
    #     #     preprocessing.normalize(X, norm='l1')
    #     # elif normalization == 'l2':
    #     #     preprocessing.normalize(X, norm='l2')
    #
    #     # features.append(X)
    #     labels.append(action_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breakfast hdf5 dataset from feature files.')

    parser.add_argument(
        '-d',
        '--features-data',
        type=str,
        dest='features_data',
        help=
        'Directory where features are stored (default: %(default)s)')

    parser.add_argument(
        '-i',
        '--videos-info',
        type=str,
        dest='videos_info',
        help=
        'File (json) where info of the videos is stored (default: %(default)s)')

    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        dest='labels',
        help=
        'File (txt) where labels are listed (default: %(default)s)')

    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        dest='output_dir',
        help=
        'Directory where hd5 file will be generated (default: %(default)s)')

    args = parser.parse_args()

    create(args.features_data, args.videos_info, args.labels, args.output_dir)