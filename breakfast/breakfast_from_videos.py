import json
import numpy as np
import h5py
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


def create(info_file, labels_file, max_length, frame_size, output_dir):
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

    try:
        makedirs(output_dir)
    except:
        pass

    if len(frame_size) == 1:
        height = width = frame_size
    else:
        height, width = frame_size[:2]

    subsets = ['training', 'testing', 'validation']
    for subset in subsets:
        videos = [
            key for key in videos_data.keys() if videos_data[key]['subset'] == subset
        ]

        nb_videos = len(videos)
        print('Number of videos for {} subset: {}'.format(subset, nb_videos))

        output_file = join(output_dir, subset + '.h5')
        f_dataset = h5py.File(output_file, 'w')

        f_dataset.create_dataset('video_features',(len(videos),max_length,height,width,3), fillvalue=0,
                            dtype=np.uint8, chunks=(1,max_length//100,height,width,3), compression='gzip', compression_opts=9)
        f_dataset.create_dataset('outputs',(len(videos),max_length), fillvalue=255,
                            dtype=np.uint8, chunks=(1,100), compression='gzip', compression_opts=9)
        f_dataset.create_dataset('lengths',(len(videos),1), fillvalue=0,
                            dtype=np.uint8)

        class_counts = np.zeros((len(labels),), dtype=np.float32)
        perm = np.random.RandomState(42).permutation(len(videos))

        for i,key in enumerate(videos):
            print('Reading %d/%d video and writing HDF5 file...' % (i,nb_videos-1))
            st_time = time.time()

            x = vid_to_array(videos_data[key]['url'], frame_size=(height,width))
            y = generate_output(videos_data[key], labels, length=1)

            f_dataset['video_features'][perm[i],:x.shape[0],:] = x.reshape([-1,height,width,3])
            f_dataset['outputs'][perm[i],:len(y)] = y
            f_dataset['lengths'][perm[i],0] = len(y)

            if subset == 'training':  # count to assign class weights later
                ids, counts = np.unique(y, return_counts=True)
                for id,c in zip(ids,counts): class_counts[id] += c

            print('Took %d secs.' % (time.time() - st_time))

        f_dataset.create_dataset('class_weights', data=(np.max(class_counts)/class_counts))

        # Save some additional attributes
        f_dataset.attrs['no_classes'] = len(labels)

        f_dataset.close()

        # Sanity check
        f_dataset = h5py.File(output_file, 'r')
        assert f_dataset.attrs['no_classes'] == len(labels)
        f_dataset.close()


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
        '-m',
        '--max-length',
        type=int,
        dest='max_length',
        default=10000,
        help=
        'Max length of sequences (default: %(default)s)')

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
        default='breakfast/dataset/',
        help=
        'Directory where 3 hd5 files will be generated (training / validation / testing) (default: %(default)s)')

    args = parser.parse_args()
    print args

    create(args.videos_info,
           args.labels,
           args.max_length,
           args.frame_size,
           args.output_dir)