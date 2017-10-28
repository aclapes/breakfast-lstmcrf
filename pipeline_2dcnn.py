import argparse
import exceptions
import os

import h5py
import numpy as np
import progressbar
import tensorflow as tf

import src.vgg19_trainable as vgg19
from src.preprocessing import compute_class_weights

np.set_printoptions(precision=2, linewidth=150)
tf.logging.set_verbosity(tf.logging.ERROR)

class Simple2DCnnModel(object):
    def __init__(self, config, input_data, is_training, pretrain_weights=None):
        self.config = config
        self.input_data = input_data
        self.is_training = is_training
        self.pretrain_weights = pretrain_weights

    def build(self):
        no_instances = self.config['no_instances']
        no_classes = self.config['no_classes']
        batch_size = self.config['batch_size']
        frame_dims = self.config['frame_dims']
        # optimizer_type = config['optimizer_type']
        decay_rate = self.config['decay_rate']
        learn_rate = self.config['learn_rate']
        # clip_norm = config['clip_norm']
        #
        self.class_weights = self.config['class_weights']

        n_gpus = len(str.split(os.environ['CUDA_VISIBLE_DEVICES'], ','))

        #
        # # Graph construction
        #

        def gen():
            """ A simple data iterator """
            n =  self.input_data.shape[0]
            perm = np.random.permutation(n)
            for i in range(n):
                yield (self.input_data[perm[i]]['image'].astype(np.float32),  self.input_data[perm[i]]['label'])
            return

        def preprocessing(x,y):
            # Step: random crops
            # Description: half of them HxW, and the others (H/2)x(W/2) scaled back to HxW.
            height = width = 224
            x_first, x_last = tf.split(x, num_or_size_splits=2, axis=0)

            x_big = tf.image.extract_glimpse(x_first,
                                             [height, width],
                                             tf.random_uniform([batch_size//2, 2], minval=-1, maxval=1))
            x_small = tf.image.extract_glimpse(x_last,
                                               [height//2, width//2],
                                               tf.random_uniform([batch_size//2, 2], minval=-1, maxval=1))

            x_small = tf.image.resize_images(x_small,
                                             [height, width],
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                             align_corners=False)

            x = tf.concat([x_big, x_small], axis=0)

            # Step: ...

            return x,y

        with tf.device('/cpu:0'):
            ds = (
                tf.data.Dataset.from_generator(gen,
                                               (tf.float32, tf.int32),
                                               (tf.TensorShape([None,None,3]), tf.TensorShape([])))
                .batch(batch_size)
                .map(lambda x, y: preprocessing(x,y))
                .prefetch(8)
            )

            self.iterator = ds.make_initializable_iterator()

            x_batch, y_batch = self.iterator.get_next()


        # # Features, output labels, and binary mask of valid timesteps
        # self.x_batch = tf.placeholder(tf.float32, shape=[None] + list(frame_dims))
        # self.y_batch = tf.placeholder(tf.int32, shape=[None])

        # # Using tftables
        # with tf.name_scope('input'):
        #     with tf.device("/cpu:0"):
        #         array_batch_placeholder = self.reader.get_batch(
        #             path = '/dataset',
        #             cyclic = False,
        #             ordered = False
        #         )
        #         x_batch_cpu = tf.to_float(array_batch_placeholder['image'])
        #         y_batch_cpu = array_batch_placeholder['label']
        #
        #         self.loader = self.reader.get_fifoloader(queue_size=1, inputs=[x_batch_cpu, y_batch_cpu], threads=1)
        #
        # x_batch, y_batch = self.loader.dequeue()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        boundaries = (np.array([10, 100, 1000], dtype=np.float32) * batch_size).tolist()
        values = [learn_rate/(decay_rate**i) for i in range(len(boundaries)+1)]
        curr_learn_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        optimizer = tf.train.AdamOptimizer(learning_rate=curr_learn_rate)

        m = batch_size//n_gpus
        tower_grads = []
        losses = []
        preds = []
        for g in range(n_gpus):
            reuse_param = (True if tf.get_variable_scope().reuse or g > 0 else False)
            with tf.device('/gpu:%d' % g):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_param):
                    # input to cnn
                    vgg = vgg19.Vgg19(self.pretrain_weights, fc8_dict=dict(size=no_classes, name='fc8'))
                    vgg.build(x_batch[g*m:(g+1)*m] / 255.0, train_mode=tf.constant(self.is_training, dtype=tf.bool))
                    # vgg.build(x_batch / 255.0, train_mode=tf.constant(self.is_training, dtype=tf.bool))

                    # output from cnn
                    logits = vgg.prob

                    preds.append(tf.argmax(logits, 1, output_type=tf.int32))

                    # Class weighting
                    y_onehot = tf.one_hot(y_batch[g*m:(g+1)*m], no_classes, on_value=1.0, off_value=0.0, axis=-1)
                    # y_onehot = tf.one_hot(y_batch, no_classes, on_value=1.0, off_value=0.0, axis=-1)

                    classweights = tf.expand_dims(tf.constant(self.class_weights, dtype=y_onehot.dtype), axis=0)
                    weight_per_label = tf.transpose(tf.matmul(y_onehot, tf.transpose(classweights)))
                    xent = tf.multiply(weight_per_label,
                                       tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_onehot))

                    # compute loss and framewise predictions
                    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss = tf.reduce_mean(xent)  # + tf.add_n(reg_losses)
                    losses.append(loss)

                    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.config['tvar_names'])
                    grads = optimizer.compute_gradients(loss, tvars)
                    tower_grads.append(grads)

        self.loss = tf.reduce_mean(losses)

        preds = tf.concat(preds, axis=0)
        equality = tf.equal(preds, y_batch)
        self.acc = tf.reduce_mean(tf.cast(equality, tf.float32))

        def average_gradients(tower_grads):
            """Calculate the average gradient for each shared variable across all towers.
            Note that this function provides a synchronization point across all towers.
            Args:
              tower_grads: List of lists of (gradient, variable) tuples. The outer list
                is over individual gradients. The inner list is over the gradient
                calculation for each tower.
            Returns:
               List of pairs of (gradient, variable) where the gradient has been averaged
               across all towers.
            """
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)

            return average_grads

        grads = average_gradients(tower_grads)
        self.train_op = optimizer.apply_gradients(grads, global_step=global_step)


    # def read_image_feeder(self):
    #     """ A simple data iterator """
    #
    #     images = self.input_data['image']
    #     labels = self.input_data['label']
    #     batch_size = self.config['batch_size']
    #
    #     n = images.shape[0]
    #
    #     while True:
    #         perm = np.random.permutation(n)
    #         for batch_idx in range(0, n - batch_size, batch_size):
    #             images_batch = [images[perm[batch_idx + ptr]] for ptr in range(batch_size)]
    #             labels_batch = [labels[perm[batch_idx + ptr]] for ptr in range(batch_size)]
    #             yield images_batch, labels_batch


    def run_epoch(self, session):
        '''
        Iterate over all batches.
        :param session:
        :param train_op:
        :return:
        '''

        # self.reader = read_image_generator(self.input_data['video_features'],
        #                                    self.input_data['outputs'],
        #                                    batch_size=self.config['batch_size'])

        # num_instances = self.reader['video_features'].shape[0]
        # num_batches = int(np.ceil(num_instances / float(self.config['batch_size'])))
        # num_batches = int(num_instances // float(self.config['batch_size']))
        batch_loss = []
        batch_accs = []

        # hit_classes = np.zeros((len(self.class_weights),), dtype=np.float32)
        # true_classes = np.zeros((len(self.class_weights),), dtype=np.float32)

        num_batches = int(np.ceil(self.config['no_instances'] // self.config['batch_size']))

        session.run(self.iterator.initializer)

        try:
            bar = progressbar.ProgressBar(max_value=num_batches)
            for batch_i in range(num_batches):
                # batch = .reader.next()
                bar.update(batch_i)

                fetches = {
                    'loss' : self.loss,
                    'acc' : self.acc,
                }
                if self.is_training:
                    fetches['train_op'] = self.train_op

                vals = session.run(
                    fetches #,
                    # feed_dict={self.x_batch: batch[0], self.y_batch: batch[1]}
                )
                # print('[Iteration %d/%d]: loss=%.5f, acc=%2.2f%%' % (batch_i+1, num_batches, vals['loss'], 100.0*vals['acc']))

                batch_loss.append(vals['loss'])
                batch_accs.append(vals['acc'])
                # batch_accs[b] = compute_accuracy(vals['predictions'], batch[1])
                # hits, trues = foo(vals['predictions'], batch[1], self.class_weights)
                # hit_classes += hits
                # true_classes += trues
                batch_i += 1
            bar.finish()
        except tf.errors.OutOfRangeError, e:
            pass
        except exceptions.StopIteration, e:
            pass

        mean_loss = np.mean(batch_loss)
        mean_acc = np.mean(batch_accs)
        print('[Total epoch] loss=%.5f, acc=%2.2f%%' % (mean_loss, 100.0*mean_acc))

        return (mean_loss, mean_acc) #, 100*(hit_classes/true_classes)


class Simple2DCnnPipeline(object):
    def __init__(self,
                 train_file,
                 val_file,
                 te_file,
                 class_weights_file,
                 batch_size,
                 learn_rate,
                 decay_rate,
                 num_epochs,
                 drop_prob,
                 tvar_names,
                 optimizer_type='adam',
                 clip_norm=1.0):

        train = h5py.File(train_file, 'r')
        val = h5py.File(val_file, 'r')
        test = h5py.File(te_file, 'r')

        self.num_epochs = num_epochs

        train_config = dict(
            no_instances = train['/dataset'].shape[0],
            batch_size = batch_size,
            frame_dims = vgg19.VGG_INPUT,
            optimizer_type = optimizer_type,
            drop_prob = drop_prob,
            decay_rate = decay_rate,
            learn_rate = learn_rate,
            tvar_names = tvar_names,
            clip_norm = clip_norm
        )

        # class weights to alleviate class imbalance during classification
        try:
            class_weights = np.load(class_weights_file)
        except IOError, e:
            class_weights = compute_class_weights(train, train_config['batch_size'])
            np.save(class_weights_file, class_weights)

        # create configurations

        train_config['class_weights'] = class_weights
        train_config['no_classes'] = len(class_weights)

        val_config = train_config.copy()
        val_config['num_instances'] = val['/dataset'].shape[0]
        val_config['batch_size'] = train_config['batch_size']//2

        test_config = train_config.copy()
        test_config['num_instances'] = test['/dataset'].shape[0]
        test_config['batch_size'] = 1

        self.sorting = np.argsort(class_weights)  # using class weight criterion

        # load pre-trained weights and remove output layer (which has de #classes from ImageNet)
        pretrain_weights = np.load('/data/datasets/vgg19.npy', encoding='latin1').item()
        if 'fc8' in pretrain_weights:
            del pretrain_weights['fc8']

        self.graph = tf.Graph()
        with self.graph.as_default():
            # initializer = tf.random_uniform_initializer(-0.01, 0.01)
            with tf.variable_scope('Model', reuse=False): #, initializer=initializer):
                self.train_model = Simple2DCnnModel(config=train_config, input_data=train['/dataset'], is_training=True, pretrain_weights=pretrain_weights)
                self.train_model.build()

            with tf.variable_scope('Model', reuse=True): #, initializer=initializer):
                self.val_model = Simple2DCnnModel(config=val_config, input_data=val['/dataset'], is_training=False, pretrain_weights=pretrain_weights)
                self.val_model.build()

            # with tf.variable_scope('Model', reuse=True): #, initializer=initializer):
            #     self.te_model = Simple2DCnnModel(config=test_config, input_data=te, is_training=False, pretrain_weights=pretrain_weights)

            self.init_op = tf.global_variables_initializer()


    def run(self, gpu_options):
        np.set_printoptions(precision=2,linewidth=150)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(graph=self.graph, config=config) as session:
            session.run(self.init_op)
            # session.run(self.train_model.iterator.initializer)
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord, sess=session)

            train_evals = np.zeros((self.num_epochs,3), dtype=np.float32)
            val_evals = np.zeros((self.num_epochs,3), dtype=np.float32)

            # try:
            for epoch_i in range(self.num_epochs):
                print('EPOCH %d/%d' % (epoch_i+1, self.num_epochs))
                # Train step
                # try:
                    # while not coord.should_stop():

                loss_train, mof_train = self.train_model.run_epoch(session)

                # except tf.errors.OutOfRangeError:
                #     pass
                # session.run(self.train_model.iterator.initializer)
                # (loss_train, mof_train), train_class_evals = self.train_model.run_epoch(session)
                # moc_train = np.nanmean(train_class_evals)

                # Validation step
                loss_val, mof_val = self.val_model.run_epoch(session)
                # (loss_val, mof_val), val_class_evals = self.val_model.run_epoch(session)
                # moc_val = np.nanmean(val_class_evals)
                #
                # Print summary
                # print(
                #     'TRAIN (loss/mof): %.5f/%2.2f%%, VAL (loss/mof): %.5f/%2.2f%%' % (
                #         loss_train, mof_train*100., loss_val, mof_val*100.
                #     )
                # )

                # # Print summary
                # print(
                #     'TRAIN (loss/mof/moc): %.4f/%.2f%%/%.2f%%, VAL (loss/mof/moc): %.4f/%.2f%%/%.2f%%' % (
                #         loss_train, mof_train, moc_train, loss_val, mof_val, moc_val
                #     )
                # )
                #
                # # Print per-class accuracies
                # print train_class_evals[self.sorting]
                # print val_class_evals[self.sorting]

                # # Keep track of loss/mof/moc across epochs
                # train_evals[e,:] = [loss_train, mof_train, moc_train]
                # val_evals[e,:] = [loss_val, mof_val, moc_val]
                #
                # # # Train step (every few epochs). To see progress (not choosing based on this!)
                # # if e in [5, 10, 50, 100, 500, 1000, 2000, 10000, 20000]:
                # #     (loss_te, mof_te), te_class_evals = self.te_model.run_epoch(session)
                # #     moc_te = np.nanmean(te_class_evals)
                # #     print te_class_evals[self.sorting]
                # #     print('TE (mof/moc): %.2f%%/%.2f%%' % (mof_te,moc_te))
            # (_, mof_te), te_class_evals = self.te_model.run_epoch(session)
            # moc_te = np.nanmean(te_class_evals)
            # print(
            #     'TRAIN (mof/moc): %.2f%%/%.2f%%, VAL (mof/moc): %.2f%%/%.2f%%, TE (mof/moc): %.2f%%/%.2f%%' % (
            #         np.nanmean(train_evals[:,1]), np.nanmean(train_evals[:,2]),
            #         np.nanmean(val_evals[:,1]), np.nanmean(val_evals[:,2]),
            #         mof_te, moc_te)
            # )
            # finally:
            #     coord.request_stop()
            #     coord.join(threads=threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform image classification using a 2D CNN (VGG) model.')

    # -----------------------------------------------
    # General parameters
    # -----------------------------------------------
    parser.add_argument(
        '-i',
        '--input-dir',
        type=str,
        dest='input_dir',
        default='/data/datasets/breakfast/images/',
        help=
        'Dataset in hdf5 format (default: %(default)s)')

    parser.add_argument(
        '-m',
        '--channel-means-file',
        type=str,
        dest='channel_means_file',
        default='./breakfast/channel_means.npy',
        help=
        'File (npy) containing a 3-valued vector (default: %(default)s)')

    parser.add_argument(
        '-w',
        '--class-weights-file',
        type=str,
        dest='class_weights_file',
        default='./breakfast/class_weights.npy',
        help=
        'File (npy) containing a N-sized vector, where N number of classes (default: %(default)s)')

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
        default=0.01,
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
        default=20,
        help=
        'Num epochs (default: %(default)s)')

    parser.add_argument(
        '-v',
        '--trainable-vars',
        type=str,
        dest='tvar_names',
        default='Model/fc8',
        help=
        '"Model/fc" to train all FC or "Model/fc8" for only output layer (default: %(default)s)')

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
        default=0.75,
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

    args = parser.parse_args()
    print args

    # Read breakfast from hdf5 file
    # f_training = h5py.File(os.path.join(args.input_dir, 'training.h5'), 'r')
    # f_validation = h5py.File(os.path.join(args.input_dir, 'validation.h5'), 'r')
    # f_testing = h5py.File(os.path.join(args.input_dir, 'testing.h5'), 'r')

    # reader_te_data = tftables.open_file(filename=os.path.join(args.input_dir, 'testing.h5'), batch_size=args.batch_size)

    if args.cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # Create a model (choosen via argument passing)
    m = Simple2DCnnPipeline(
        os.path.join(args.input_dir, 'testing.test.h5'),
        os.path.join(args.input_dir, 'testing.h5'),
        os.path.join(args.input_dir, 'testing.h5'),
        args.class_weights_file,
        batch_size=args.batch_size,
        learn_rate=args.learn_rate,
        decay_rate=args.decay_rate,
        num_epochs=args.num_epochs,
        drop_prob=args.drop_prob,
        tvar_names=args.tvar_names,
        optimizer_type=args.optimizer_type,
        clip_norm=args.clip_norm
    )

    # -----------------------------------------------
    # RUN
    # -----------------------------------------------
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    m.run(gpu_options)
    # -----------------------------------------------

    # f_testing.close()
    # f_validation.close()
    # f_training.close()
