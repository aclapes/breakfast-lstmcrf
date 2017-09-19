import h5py
import argparse
import tensorflow as tf

from pipeline_lstm import SimpleLstmPipeline
from pipeline_lstm_ttbp import LstmPipeline
from pipeline_crf import SimpleCrfPipeline
from pipeline_lstmcrf import SimpleLstmcrfPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform labelling of sequences using a LSTMCRF model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        dest='input_file',
        default='/breakfast/breakfast.h5',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=48,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=1e-2,
        help=
        'Starting learning rate. It decays after 100 and 1000 epochs by a factor specified by --decay-rate argument (default: %(default)s)')

    parser.add_argument(
        '-dr',
        '--decay-rate',
        type=float,
        dest='decay_rate',
        default=10,
        help=
        'Decay rate for inverse time decay (default: %(default)s)')

    parser.add_argument(
        '-e',
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=2000,
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
        '-M',
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
        default=1024,
        help=
        'Hidden size (default: %(default)s)')

    parser.add_argument(
        '-p',
        '--drop-prob',
        type=float,
        dest='drop_prob',
        default=0.1,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------
    parser.add_argument(
        '-G',
        '--gpu-memory',
        type=float,
        dest='gpu_memory',
        default=0.9,
        help=
        'GPU memory to reserve (default: %(default)s)')

    args = parser.parse_args()
    print args

    # Read breakfast from hdf5 file
    f_dataset = h5py.File(args.input_file, 'r')
    print('Dataset (%s) attributes:' % (args.input_file))
    for key in f_dataset.attrs.keys():
        print('%s : %s' % (key, str(f_dataset.attrs[key])))

    # Create a model (choosen via argument passing)
    if args.model_type == 'lstmcrf':
        m = SimpleLstmcrfPipeline(
            f_dataset['training'],
            f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            f_dataset['training']['class_weights'][:],
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
        # m = SimpleLstmPipeline(
        #     f_dataset['training'],
        #     f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
        #     f_dataset['testing'],
        #     f_dataset.attrs['no_classes'],
        #     f_dataset['training']['class_weights'][:],
        #     batch_size=args.batch_size,
        #     learn_rate=args.learn_rate,
        #     decay_rate=args.decay_rate,
        #     num_epochs=args.num_epochs,
        #     hidden_size=args.hidden_size,
        #     drop_prob=args.drop_prob,
        #     optimizer_type=args.optimizer_type,
        #     clip_norm=args.clip_norm
        # )
        m = SimpleLstmPipeline(
            f_dataset['training'],
            f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            f_dataset['training']['class_weights'][:],
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
            f_dataset['training'],
            f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            f_dataset['training']['class_weights'][:],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    elif args.model_type == 'cnncrf':
        m = SimpleCnnCrfPipeline(
            f_dataset['training'],
            f_dataset['validation'] if 'validation' in f_dataset else f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            f_dataset['training']['class_weights'][:],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            decay_rate=args.decay_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type,
            clip_norm=args.clip_norm
        )
    else:
        raise NotImplementedError('Please specify a valid model (-M <model_type>).')

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    m.run(gpu_options)
    # -----------------------------------------------

    f_dataset.close()
