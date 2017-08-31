import scipy.io as sio
import numpy as np
import argparse
import h5py


def import_labels(f):
    ''' Read from a file all the labels from it '''
    lines = f.readlines()
    labels = []
    i = 0
    for l in lines:
        t = l.split(':')
        assert int(t[0]) == i
        label = t[1].split('\n')[0]
        labels.append(label)
        i += 1
    return labels


def read_train_flags(mat):
    return np.array([elem[0][8][0][0] for elem in mat['hw3']])

def read_clip_ids(mat):
    return np.array([elem[0][-1][0][0] for elem in mat['hw3']])


def create(features_file, pool_op, pool_size, stride, labels_file, output_file):
    '''

    :param features_file:
    :param pool_op:
    :param pool_size:
    :param stride:
    :param info_file:
    :param labels_file:
    :param output_file:
    :return:
    '''

    with open(labels_file, 'r') as f:
        labels = import_labels(f)

    # load dataset
    mat = sio.loadmat(features_file)

    # create: ids, features, outputs, lengths and holdout train/test flag
    video_ids = read_clip_ids(mat)
    X = np.squeeze(mat['X'])
    Y = np.array(
        [np.argmax(np.concatenate([y[:,-1][:,np.newaxis],y[:,:-1]], axis=1), axis=1)  # make the none class id = 0
         for y in np.squeeze(mat['Y'])]
    )
    lengths = np.array([y.shape[0] for y in Y])
    train_flag = read_train_flags(mat)

    num_features = X[0].shape[1]
    max_len = ((np.max(lengths) // stride) + 1) * stride

    subsets = ['training', 'testing']
    subset_inds = [np.where(train_flag==1)[0], np.where(train_flag!=1)[0]]

    f_dataset = h5py.File(output_file, 'w')

    for k,subset in enumerate(subsets):
        print('Creating HDF file for %s...' % (subset))

        inds = subset_inds[k]

        # video_subset_ids = video_ids[inds][:,np.newaxis]
        video_features = np.zeros((len(inds), max_len, num_features), dtype=np.float32)
        outputs = np.zeros((len(inds), max_len), dtype=np.float32)
        class_counts = np.zeros((len(labels),), dtype=np.float32)

        for i,idx in enumerate(inds):
            l = lengths[idx]
            video_features[i,:l,:] = X[idx]
            outputs[i,:l] = Y[idx]
            if subset == 'training':
                class_ids, counts = np.unique(Y[idx], return_counts=True)
                for id, c in zip(class_ids, counts): class_counts[id] += c

        perm = np.random.RandomState(42).permutation(len(inds))

        f_dataset_subset = f_dataset.create_group(subset)
        f_dataset_subset.create_dataset(
            'video_ids',
            data=video_ids[inds][perm,np.newaxis],
            chunks=(16, 1),
            dtype='int32')
        f_dataset_subset.create_dataset(
            'video_features',
            data=video_features[perm, :, :],
            chunks=(16, video_features.shape[1], video_features.shape[2]),
            dtype='float32')
        f_dataset_subset.create_dataset(
            'outputs',
            data=outputs[perm, :],
            chunks=(16, outputs.shape[1]),
            dtype='float32')
        f_dataset_subset.create_dataset(
            'lengths',
            data=lengths[inds][perm,np.newaxis],
            chunks=(16, 1),
            dtype='int32')

        # class in train without instances. avoid zero division
        class_weights = np.max(class_counts)/class_counts
        class_weights[class_counts == 0] = 0.
        f_dataset_subset.create_dataset('class_weights', data=class_weights)

    # Save some additional attributes
    f_dataset.attrs['no_classes'] = len(labels)
    f_dataset.attrs['pool_op'] = pool_op
    f_dataset.attrs['pool_size'] = pool_size
    f_dataset.attrs['stride'] = stride

    f_dataset.close()

    # Sanity check
    f_dataset = h5py.File(output_file, 'r')
    # f_dataset.attrs['class_weights']
    assert f_dataset.attrs['no_classes'] == len(labels)
    f_dataset.close()


    # for i,elem in enumerate(mat['hw3']):
    #     instance = dict(
    #         movie = elem[i][0][0],
    #         movid = elem[i][1][0][0],
    #         frames = elem[i][2]
    #     )
    #
    #     print instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create "hollywood extended" hdf5 dataset.')

    parser.add_argument(
        '-d',
        '--features-dir',
        type=str,
        dest='features_file',
        default='/datasets/hollywood_extended/full_dataset.mat',
        help=
        'MAT file where features are stored (default: %(default)s)')

    # parser.add_argument(
    #     '-i',
    #     '--videos-info',
    #     type=str,
    #     dest='videos_info',
    #     default='breakfast/videos.json',
    #     help=
    #     'File (json) where info of the videos is stored (default: %(default)s)')

    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        dest='labels',
        default='hollywoodext/labels.txt',
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
        default='hollywoodext/dataset.h5',
        help=
        'Directory where hd5 file will be generated (default: %(default)s)')

    args = parser.parse_args()
    print args

    create(args.features_file,
           args.pool_op,
           args.pool_size,
           args.stride,
           args.labels,
           args.output_file)

    quit()