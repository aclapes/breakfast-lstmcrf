import json
import numpy as np
import tables
import argparse
import imageio
from scipy.misc import imresize
from os.path import join
from os import makedirs
import time

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


def generate_output(video_info, labels):
    nb_frames = video_info['num_frames']

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

    instances = [labels.index(out) for out in outputs]

    return instances


def vid_to_array(filepath, frame_size=None):
    arr = None

    # init video capture
    cap = imageio.get_reader(filepath)
    # iterate over the video frames
    num_frames = cap.get_meta_data()['nframes']
    for i in range(num_frames):
        try:
            im = cap.get_next_data()  # obtain the frame

            # determine final image size size
            if frame_size is None:
                height, width, _ = im.shape
            else:
                height, width = frame_size[0], frame_size[1]

            # now that we know frame dimensions, init arr to store the frames
            if arr is None:
                arr = np.empty((num_frames, height, width, im.shape[-1]), dtype=np.uint8)

            arr[i] = im if (frame_size is None) else imresize(im, size=(height,width), interp='bilinear')
        except:
            arr[i] = np.zeros((height,width,im.shape[-1]), dtype=np.uint8)
    cap.close()

    return arr


def get_num_frames(filepath):
    # init video capture
    cap = imageio.get_reader(filepath)
    # iterate over the video frames
    n_frames = cap.get_meta_data()['nframes']
    cap.close()

    return n_frames

def next_multiple_of(x, y):
    """
    Get next multiple of y after x.
    :param x:
    :param y:
    :return:
    """
    return y * (x // y) + y


class TableRow(tables.IsDescription):
    image = tables.UInt8Col(shape=(224,224,3))
    label = tables.Int32Col()

def create(info_file, labels_file, frame_size, output_dir, rate_frames=0.20):
    with open(info_file, 'r') as f:
        videos_data = json.load(f)
        # uncomment if want to merge validation and testing
        # ---
        for key, value in videos_data.iteritems():
            if videos_data[key]['subset'] == 'validation':
                videos_data[key]['subset'] = 'training'
        # ---

    with open(labels_file, 'r') as f:
        labels = import_labels(f)

    try:
        makedirs(output_dir)
    except:
        pass

    if len(frame_size) == 1:
        height = width = frame_size
    else:
        height, width = frame_size[:2]

    subsets = ['testing', 'training']
    for subset in subsets:
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]

        nb_videos = len(videos)
        print('Number of videos for {} subset: {}'.format(subset, nb_videos))

        nr_samples = 0
        for key in videos:
            nr_samples += int(videos_data[key]['num_frames']*rate_frames)

        output_file = join(output_dir, subset + '.blosc.h5table')
        filters = tables.Filters(complevel=9, complib='blosc')
        file = tables.open_file(output_file, "w", filters=filters)

        # f_dataset.create_array(f_dataset.root, 'video_features',
        #                        tables.Atom.from_dtype(sample_data.dtype)
        #                        (nr_samples,height,width,3), chunks=(1,height,width,3), dtype=np.uint8, compression='gzip', compression_opts=9)
        table = file.create_table(file.root, 'dataset', TableRow, expectedrows=nr_samples, chunkshape=[1])

        # f_dataset.create_dataset('video_features',(nr_samples,height,width,3), dtype=np.uint8)
        # f_dataset.create_dataset('outputs', (nr_samples,), dtype=np.uint8)

        class_counts = np.zeros((len(labels),), dtype=np.float32)
        perm_gbl = np.random.RandomState(42).permutation(nr_samples)

        ptr = 0
        row = table.row
        for i,key in enumerate(videos):
            print('Reading %d/%d video and writing HDF5 file...' % (i,nb_videos-1))
            st_time = time.time()
            X = vid_to_array(videos_data[key]['url'], frame_size=(height,width))
            Y = np.array(generate_output(videos_data[key], labels))
            assert X.shape[0] == len(Y) == videos_data[key]['num_frames']
            print('Reading took %d secs.' % (time.time() - st_time))

            st_time = time.time()

            for k, (x,y) in enumerate(zip(X,Y)):
                row['image'] = x
                row['label'] = y
                row.append()

            # assert X.shape[0] == len(Y)
            # f_dataset['video_features'][gbl_perm[ptr:ptr+len(perm_subset)]] = X[perm_subset]
            # f_dataset['outputs'][gbl_perm[ptr:ptr+len(perm_subset)]] = Y[perm_subset]
            # ptr += len(perm_subset)

            # for k,(x,y) in enumerate(zip(X,Y)):
            #     # f_dataset['video_features'][perm[ptr]] = x
            #     # f_dataset['outputs'][perm[ptr]] = y
            #     f_dataset['video_features'][ptr] = x
            #     f_dataset['outputs'][ptr] = y
            #     class_counts[y] += 1
            #     ptr += 1

            # perm_vid = np.random.RandomState(42).permutation(videos_data[key]['num_frames'])
            # perm_vid_sb = perm_vid[:int(len(perm_vid)*rate_frames)]
            #
            # perm_gbl_sb = perm_gbl[ptr:ptr+len(perm_vid_sb)]

            # for idx_gbl_sb,idx_vid_sb in zip(perm_gbl_sb, perm_vid_sb):
            #     f_dataset['video_features'][idx_gbl_sb] = X[idx_vid_sb]
            #     f_dataset['outputs'][idx_gbl_sb] = Y[idx_vid_sb]
            #     class_counts[Y[idx_vid_sb]] += 1
            # ptr += len(perm_vid_sb)

            # # subsample
            # step = frameskip
            # end = step * (len(y)//step)
            # x_s, y_s = x[:end:step], y[:end:step]
            #
            # for k,(im,lbl) in enumerate(zip(x_s,y_s)):
            #     f_dataset['video_features'][perm[ptr]] = im
            #     f_dataset['outputs'][perm[ptr]] = lbl
            #     class_counts[lbl] += 1
            #     ptr += 1

            print('Writing to HDF5 took %d secs.' % (time.time() - st_time))
        table.flush()

        assert ptr == nr_samples

        # f_dataset.create_dataset('class_weights', data=(np.max(class_counts)/class_counts))

        # Save some additional attributes
        # f_dataset.attrs['no_classes'] = len(labels)

        # f_dataset.close()

        # Sanity check
        # f_dataset = h5py.File(output_file, 'r')
        # assert f_dataset.attrs['no_classes'] == len(labels)
        # f_dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create breakfast hdf5 breakfast from feature files.')

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
        '-ws',
        '--win-size',
        type=int,
        dest='win_size',
        default=16,
        help=
        'Window size (default: %(default)s)')

    parser.add_argument(
        '-s',
        '--frame-size',
        nargs='+',
        type=int,
        dest='frame_size',
        default=[224,224],
        help=
        'Resize frames to (default: %(default)s)')

    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        dest='output_dir',
        default='/data/datasets/breakfast/images/',
        help=
        'Directory where 3 hdf5 files will be generated (training / validation / testing) (default: %(default)s)')

    args = parser.parse_args()
    print args

    create(args.videos_info,
           args.labels,
           args.frame_size,
           args.output_dir)