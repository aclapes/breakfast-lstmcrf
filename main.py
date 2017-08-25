import h5py
import argparse

from crf import CrfPipeline
from lstm import LstmPipeline
from lstmcrf import LstmCrfPipeline

from simple_lstm import SimpleLstmPipeline # debug

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
        default='/data/datasets/breakfast/fv/s1/dataset.8-20.h5',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=32,
        help=
        'Batch size (default: %(default)s)')

    parser.add_argument(
        '-lr',
        '--learning-rate',
        type=float,
        dest='learn_rate',
        default=1e-4,
        help=
        'Learning rate (default: %(default)s)')

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
        default=512,
        help=
        'Hidden size (default: %(default)s)')

    parser.add_argument(
        '-p',
        '--drop-prob',
        type=float,
        dest='drop_prob',
        default=0.5,
        help=
        'Dropout probability (default: %(default)s)')
    # -----------------------------------------------

    args = parser.parse_args()
    print args

    # Read dataset from hdf5 file
    f_dataset = h5py.File(args.input_file, 'r')
    print('Dataset (%s) attributes:' % (args.input_file))
    for key in f_dataset.attrs.keys():
        print('%s : %s' % (key, str(f_dataset.attrs[key])))

    m = SimpleLstmPipeline(
        f_dataset['training'],
        f_dataset['validation'],
        f_dataset['testing'],
        f_dataset.attrs['no_classes'],
        batch_size=args.batch_size,
        learn_rate=args.learn_rate,
        num_epochs=args.num_epochs,
        hidden_size=args.hidden_size,
        drop_prob=args.drop_prob,
        optimizer_type=args.optimizer_type
    )
    m.run()
    quit()

    # Create a model (choosen via argument passing)
    if args.model_type == 'lstmcrf':
        m = LstmCrfPipeline(
            f_dataset['training'],
            f_dataset['validation'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type
        )
    elif args.model_type == 'lstm':
        m = LstmPipeline(
            f_dataset['training'],
            f_dataset['testing'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            hidden_size=args.hidden_size,
            drop_prob=args.drop_prob,
            optimizer_type=args.optimizer_type
        )
    elif args.model_type == 'crf':
        m = CrfPipeline(
            f_dataset['training'],
            f_dataset['validation'],
            f_dataset['testing'],
            f_dataset.attrs['no_classes'],
            batch_size=args.batch_size,
            learn_rate=args.learn_rate,
            num_epochs=args.num_epochs,
            optimizer_type=args.optimizer_type
        )
    else:
        raise NotImplementedError('Please specify a valid model (-M <model_type>).')

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    m.run()
    # -----------------------------------------------

    f_dataset.close()
