from __future__ import print_function
import os
import re
import csv
import numpy as np
# from sklearn import preprocessing
from os.path import basename, splitext, join
import imageio
# from skimage.transform import resize
from imageio.core.format import CannotReadFrameError
import h5py
import random
import imageio


def create_json_and_labels(path_videos, path_segmentation, output_labels_file, output_json_file):
    config = dict()
    # Define train/test split
    # config['pattern_train'] = '(P16_|P17_|P18_|P19_|P20_|P21_|P22_|P23_|P24_|P25_|P26_|P27_|P28_|P29_|P30_|P31_|P32_|P33_|P34_|P35_|P36_|P37_|P38_|P39_|P40_|P41_|P42_|P43_|P44_|P45_|P46_|P47_|P48_|P49_|P50_|P51_|P52_|P53_|P54_)'
    config['pattern_test'] = '(P03_|P04_|P05_|P06_|P07_|P08_|P09_|P10_|P11_|P12_|P13_|P14_|P15_)'
    config['pattern_train'] = '(P27_|P28_|P29_|P30_|P31_|P32_|P33_|P34_|P35_|P36_|P37_|P38_|P39_|P40_|P41_|P42_|P43_|P44_|P45_|P46_|P47_|P48_|P49_|P50_|P51_|P52_|P53_|P54_)'
    config['pattern_val'] =  '(P16_|P17_|P18_|P19_|P20_|P21_|P22_|P23_|P24_|P25_|P26_)'

    config['segm_file_fmt'] = '.xml'

    json_content = dict()

    action_labels = dict()  # several actions in each video

    video_labels = np.sort(os.listdir(path_segmentation))  # video labels correspond to activities, not actions
    for i, video_label in enumerate(video_labels):
        segm_files = os.listdir(os.path.join(path_segmentation, video_label))
        for segm_file in segm_files:
            if segm_file.endswith(config['segm_file_fmt']):
                # get video filename from segmentation file
                video_filename = os.path.splitext(segm_file)[0]

                # find corresponding video file path
                for fmt in ['avi', 'mp4', 'mov', 'mpeg', 'mkv']:
                    url = os.path.join(path_videos, video_filename + '.' + fmt)
                    url = url.replace('salat', 'salad')
                    url = url.replace('cereals', 'cereal')
                    if os.path.exists(url):
                        break

                # get data partition
                sid = video_filename.split('_')[0]
                if sid in config['pattern_train']:
                    subset = 'training'
                elif sid in config['pattern_val']:
                    subset = 'validation'
                else:
                    subset = 'testing'

                # gather the rest of video metadata
                video_md = imageio.get_reader(url).get_meta_data()
                nframes = video_md['nframes']
                duration = video_md['duration']
                size = video_md['size']

                # create the summary
                json_content[video_filename] = dict(
                    subset=subset,
                    num_frames=nframes,
                    url=url,
                    duration=duration,
                    resolution=(str(size[0]) + 'x' + str(size[1])),
                    annotations=[]
                )

                # process all the annotations for the video
                with open(os.path.join(path_segmentation, video_label, segm_file), 'r') as f:
                    lines = f.readlines()[2:-2]
                    for line in lines:
                        name, st, end = re.findall('"([^"]*)"', line)
                        # build annotation
                        label = name if name != 'SIL' else 'none'
                        json_content[video_filename]['annotations'].append(
                            dict(segment=[(float(st)-1)/nframes * duration,
                                          (float(end)-1)/nframes * duration],
                                 label=label)
                        )
                        # keep track of action labels
                        action_labels.setdefault(label,None)

    with open(output_labels_file, 'w') as f:
        del action_labels['none']
        f.write('0\tnone\n')
        for i,label in enumerate(action_labels.keys()):
            f.write('{}\t{}\n'.format(i+1,label))

    with open(output_json_file, 'w') as f:
        import json
        json.dump(json_content, f, indent=2)

    return


if __name__ == "__main__":
    create_json_and_labels('/data/vid/PALL/', '/data/segmentation_coarse/', 'dataset/labels.txt', 'dataset/videos.json')

    # data = load_features(config['features'], config['segmentation'], config['pattern_train'],shuffle=True,seed=42,p=1.0)

    quit()