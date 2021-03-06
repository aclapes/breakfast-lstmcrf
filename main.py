import h5py
import argparse
import tensorflow as tf

from pipeline_lstm import SimpleLstmPipeline
from pipeline_crf import SimpleCrfPipeline
from pipeline_scrf import SimpleScrfPipeline
from pipeline_lstmcrf import SimpleLstmcrfPipeline
from pipeline_lstmscrf import SimpleLstmScrfPipeline


import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform labelling of sequences using a LSTMCRF model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='/data/hupba/Datasets/breakfast/hdf5/pooled-20-0c/',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=64,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=1e-3,
        help=
        'Starting learning rate. It decays after 100 and 1000 epochs by a factor specified by --decay-rate argument (default: %(default)s)')

    parser.add_argument(
        '-dr',
        '--decay-rate',
        type=float,
        dest='decay_rate',
        default=2,
        help=
        'Decay rate for inverse time decay (default: %(default)s)')

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=150,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-ot',
        '--optimizer-type',
        type=str,
        dest='optimizer_type',
        default='adam',
        help=
        'Optimizer type (sgd or adam) (default: %(default)s)')

    parser.add_argument(
        '-c',
        '--clip-norm',
        type=float,
        dest='clip_norm',
        default=5.0,
        help=
        'Clipping gradients by norm above clip_norm (default: %(default)s)')

    parser.add_argument(
        '-T',
        '--test-subset',
        type=str,
        dest='test_subset',
        default='s1',
        help=
        'Test data subset identifier (default: %(default)s)')

    parser.add_argument(
        '-m',
        '--model-type',
        type=str,
        dest='model_type',
        default='lstmcrf',
        help=
        'Model type (crf, lstm or lstmcrf) (default: %(default)s)')

    # -----------------------------------------------
    # (LSTM-only parameters)
    # -----------------------------------------------
    parser.add_argument(
        '-s',
        '--hidden-size',
        type=int,
        dest='hidden_size',
        default=128,
        help=
        'Hidden size (default: %(default)s)')

    parser.add_argument(
        '-p',
        '--drop-prob',
        type=float,
        dest='drop_prob',
        default=0.2,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------
    parser.add_argument(
        '-G',
        '--gpu-memory',
        type=float,
        dest='gpu_memory',
        default=0.95,
        help=
        'GPU memory to reserve (default: %(default)s)')

    parser.add_argument(
        '-D',
        '--cuda-devices',
        type=str,
        dest='cuda_devices',
        default="3",
        help=
        'GPU devices (default: %(default)s)')

    parser.add_argument(
        '-L',
        '--logging-path',
        type=str,
        dest='logging_path',
        default='/data/hupba/Datasets/breakfast/log/',
        help=
        'Tensorboard\'s logging path (default: %(default)s)')

    parser.add_argument(
        '-M',
        '--models-path',
        type=str,
        dest='models_path',
        default='/data/hupba/Datasets/breakfast/models/',
        help=
        'Tensorflow\'s models path (default: %(default)s)')

    args = parser.parse_args()
    print args

    # Read breakfast from hdf5 file
    f_dataset = h5py.File(os.path.join(args.input_dir, 'dataset.h5'), 'r')

    # f_training = h5py.File(os.path.join(args.input_dir, 'training.h5'), 'r')
    # f_validation = h5py.File(os.path.join(args.input_dir, 'testing.h5'), 'r')
    # f_testing = h5py.File(os.path.join(args.input_dir, 'testing.h5'), 'r')


    # # Read breakfast from hdf5 file
    # f_dataset = h5py.File(args.input_file, 'r')
    # print('Dataset (%s) attributes:' % (args.input_file))
    # for key in f_dataset.attrs.keys():
    #     print('%s : %s' % (key, str(f_dataset.attrs[key])))

    if args.cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # Create a model (choosen via argument passing)
    if args.model_type == 'lstmcrf':
        m = SimpleLstmcrfPipeline(
            f_dataset,
            args.test_subset,
            os.path.join(args.logging_path, 'lstmcrf', args.test_subset),
            os.path.join(args.models_path, 'lstmcrf', args.test_subset),
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    elif args.model_type == 'lstm':
        m = SimpleLstmPipeline(
            f_dataset,
            args.test_subset,
            os.path.join(args.logging_path, 'lstm', args.test_subset),
            os.path.join(args.models_path, 'lstm', args.test_subset),
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    elif args.model_type == 'crf':
        m = SimpleCrfPipeline(
            f_dataset,
            args.test_subset,
            os.path.join(args.logging_path, 'crf', args.test_subset),
            os.path.join(args.models_path, 'crf', args.test_subset),
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    elif args.model_type == 'scrf':
        m = SimpleScrfPipeline(
            f_dataset,
            args.test_subset,
            os.path.join(args.logging_path, 'scrf', args.test_subset),
            os.path.join(args.models_path, 'scrf', args.test_subset),
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    elif args.model_type == 'lstmscrf':
        m = SimpleLstmScrfPipeline(
            f_dataset,
            args.test_subset,
            os.path.join(args.logging_path, 'lstmscrf', args.test_subset),
            os.path.join(args.models_path, 'lstmscrf', args.test_subset),
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    # elif args.model_type == 'lstm':
    #     m = SimpleLstmPipeline(
    #         f_training,
    #         f_testing,
    #         f_testing,
    #         args.class_weights_file,
    #         batch_size=args.batch_size,
    #         learn_rate=args.learn_rate,
    #         decay_rate=args.decay_rate,
    #         num_epochs=args.num_epochs,
    #         hidden_size=args.hidden_size,
    #         drop_prob=args.drop_prob,
    #         optimizer_type=args.optimizer_type,
    #         clip_norm=args.clip_norm
    #     )
    # elif args.model_type == 'crf':
    #     m = SimpleCrfPipeline(
    #         f_dataset['training'],
    #         f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
    #         f_dataset['testing'],
    #         f_dataset.attrs['no_classes'],
    #         f_dataset['training']['class_weights'][:],
    #         batch_size=args.batch_size,
    #         learn_rate=args.learn_rate,
    #         decay_rate=args.decay_rate,
    #         num_epochs=args.num_epochs,
    #         hidden_size=args.hidden_size,
    #         drop_prob=args.drop_prob,
    #         optimizer_type=args.optimizer_type,
    #         clip_norm=args.clip_norm
    #     )
    # elif args.model_type == 'cnncrf':
    #     m = SimpleCnnCrfPipeline(
    #         f_dataset['training'],
    #         f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
    #         f_dataset['testing'],
    #         f_dataset.attrs['no_classes'],
    #         f_dataset['training']['class_weights'][:],
    #         batch_size=args.batch_size,
    #         learn_rate=args.learn_rate,
    #         decay_rate=args.decay_rate,
    #         num_epochs=args.num_epochs,
    #         hidden_size=args.hidden_size,
    #         drop_prob=args.drop_prob,
    #         optimizer_type=args.optimizer_type,
    #         clip_norm=args.clip_norm
    #     )
    else:
        raise NotImplementedError('Please specify a valid model (-M <model_type>).')

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    m.run(gpu_options)
    # -----------------------------------------------

    # f_training.close()
    # f_validation.close()
    # f_testing.close()
