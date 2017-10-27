from __future__ import print_function
import os
import re
import numpy as np
import imageio
import argparse


def create_json_and_labels(path_videos, path_segmentation, output_labels_file, output_json_file):
    config = dict()

    # Define test split:
    config['pattern_test'] = '(P03_|P04_|P05_|P06_|P07_|P08_|P09_|P10_|P11_|P12_|P13_|P14_|P15_)'

    # Define training (and validation). Decide whether or not to have a validation split:
    # </---
    # config['pattern_train'] = '(P16_|P17_|P18_|P19_|P20_|P21_|P22_|P23_|P24_|P25_|P26_|P27_|P28_|P29_|P30_|P31_|P32_|P33_|P34_|P35_|P36_|P37_|P38_|P39_|P40_|P41_|P42_|P43_|P44_|P45_|P46_|P47_|P48_|P49_|P50_|P51_|P52_|P53_|P54_)'
    # ---
    config['pattern_train'] = '(P27_|P28_|P29_|P30_|P31_|P32_|P33_|P34_|P35_|P36_|P37_|P38_|P39_|P40_|P41_|P42_|P43_|P44_|P45_|P46_|P47_|P48_|P49_|P50_|P51_|P52_|P53_|P54_)'
    config['pattern_val'] =  '(P16_|P17_|P18_|P19_|P20_|P21_|P22_|P23_|P24_|P25_|P26_)'
    # ---/>

    config['segm_file_fmt'] = '.xml'

    json_content = dict()

    action_labels = dict()  # several actions in each video

    action_durations = dict()

    video_labels = np.sort(os.listdir(path_segmentation))  # video labels correspond to activities, not actions
    for i, video_label in enumerate(video_labels):
        segm_files = os.listdir(os.path.join(path_segmentation, video_label))
        for segm_file in segm_files:
            if segm_file.endswith(config['segm_file_fmt']):
                # print(segm_file)
                # if segm_file == 'P30_cam02_P30_cereals.xml':
                #     print(1)
                # get video filename from segmentation file
                video_filename = os.path.splitext(segm_file)[0]

                # find corresponding video file path
                split = video_filename.split('_')
                url_stem = os.path.join(split[0],split[1],split[2]+'_'+split[3])
                for fmt in ['avi', 'mp4', 'mov', 'mpeg', 'mkv']:
                    url = os.path.join(path_videos, url_stem + '.' + fmt)
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
                fps = video_md['fps']
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
                        st, end = float(st)-1, float(end)-1
                        action_durations.setdefault(name, []).append(end-(st-1)+1)
                        # build annotation
                        label = name if name != 'SIL' else 'none'
                        json_content[video_filename]['annotations'].append(
                            dict(segment=[float(st)/fps, (float(end)+1)/fps],
                                 label=label)
                        )
                        # keep track of action labels
                        action_labels.setdefault(label,None)

    # not saving this, only for debugging purposes
    # ---
    for name, list_of_durations in action_durations.iteritems():
        avg = np.mean(list_of_durations)
        min = np.min(list_of_durations)
        max = np.max(list_of_durations)
        print('%s\t AVG=%.2f\t MIN=%.2f\t MAX=%.2f' % (name, avg, min, max))
    # ---

    with open(output_json_file, 'w') as f:
        import json
        json.dump(json_content, f, indent=2)

    with open(output_labels_file, 'w') as f:
        del action_labels['none']
        f.write('0\tnone\n')
        for i,label in enumerate(action_labels.keys()):
            f.write('{}\t{}\n'.format(i+1,label))

    print('Files saved')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create couple of files required to create the dataset.')

    parser.add_argument(
        '-v',
        '--videos-dir',
        type=str,
        dest='videos_dir',
        default='/data/datasets/breakfast/vid/',
        help=
        'Directory containing the breakfast data (default: %(default)s)')

    parser.add_argument(
        '-s',
        '--segmentations-dir',
        type=str,
        dest='segmentations_dir',
        default='/data/datasets/breakfast/segmentation_coarse/',
        help=
        'Directory containing the breakfast segmentation files (default: %(default)s)')

    parser.add_argument(
        '-l',
        '--labels-file',
        type=str,
        dest='labels_file',
        default='breakfast/labels.txt',
        help=
        'Output labels file (default: %(default)s)')

    parser.add_argument(
        '-i',
        '--info-file',
        type=str,
        dest='info_file',
        default='breakfast/videos.json',
        help=
        'Output info file (default: %(default)s)')

    args = parser.parse_args()
    print(args)

    create_json_and_labels(args.videos_dir,
                           args.segmentations_dir,
                           args.labels_file,
                           args.info_file)

    quit()