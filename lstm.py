import numpy as np
import tensorflow as tf

from progressbar import ProgressBar

from tensorflow.contrib import rnn

from reader import read_data_generator
from evaluation import compute_framewise_accuracy


class LstmPipeline(object):
    def __init__(self,
                 train,
                 val,
                 test,
                 no_classes,
                 batch_size,
                 learn_rate,
                 num_epochs,
                 hidden_size,
                 drop_prob,
                 optimizer_type='adam'):

        self.x, self.y, self.lengths = train['video_features'], train['outputs'], train['lengths']
        self.x_val, self.y_val, self.lengths_val = val['video_features'], val['outputs'], val['lengths']
        self.x_te, self.y_te, self.lengths_te = test['video_features'], test['outputs'], test['lengths']

        self.no_classes = no_classes

        self.batch_size = batch_size
        self.num_words = self.x.shape[1]
        self.num_features = self.x.shape[2]

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size

        self.optimizer_type = optimizer_type

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_batch = tf.placeholder(tf.float32, shape=[None, self.num_words, self.num_features])
            self.y_batch = tf.placeholder(tf.int32, shape=[None, self.num_words])
            self.w_batch = tf.placeholder(tf.float32, shape=[None, self.num_words])

            self.lengths_batch = tf.cast(tf.reduce_sum(self.w_batch, axis=1), dtype=tf.int32)

            self.state_placeholder = tf.placeholder(tf.float32, [2, None, self.hidden_size])

            self.x_batch = tf.nn.l2_normalize(self.x_batch, dim=2)
            x_drop = tf.nn.dropout(self.x_batch, keep_prob=1.0) # TODO: experiemnt with this one

            cell = rnn.LSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.drop_prob))

            init_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.state_placeholder[0], self.state_placeholder[1])
            # self.init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Obtain the idx-th from num_steps chunks of length step_size
            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(
                cell,
                x_drop,
                dtype=tf.float32,
                initial_state=init_state_tuple, # statefull rnn
                sequence_length=self.lengths_batch  # do not process padded parts
            )

            outputs_dropout = tf.nn.dropout(rnn_outputs, keep_prob=(1-self.drop_prob))

            # # Compute unary scores from a linear layer.
            # output = tf.reshape(tf.stack(self._outputs, axis=1), [-1, self.hidden_size])
            output = tf.reshape(outputs_dropout, [-1, self.hidden_size])
            softmax_w = tf.get_variable('softmax_w', [self.hidden_size, self.no_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [self.no_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
            logits = tf.matmul(output, softmax_w) + softmax_b

            normalized_logits = tf.nn.softmax(logits)
            normalized_logits = tf.reshape(
                normalized_logits, [tf.shape(self.x_batch)[0], self.num_words, self.no_classes]
            )

            self.pred = tf.argmax(normalized_logits, 2)

            loss = tf.contrib.seq2seq.sequence_loss(
                normalized_logits,
                self.y_batch,
                self.w_batch,
                average_across_timesteps=False,
                average_across_batch=True
            )
            self.cost = tf.reduce_sum(loss)

            if self.optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5.0)
            self.apply_placeholder_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()  # always session.run this op first!

    def run(self):
        with tf.Session(graph=self.graph) as session:
            session.run(self.init_op)

            for e in range(self.num_epochs):
                print('Epoch: %d/%d' % (e+1,self.num_epochs))

                self.train_reader = read_data_generator(
                    self.x, self.y, self.lengths, batch_size=self.batch_size
                )

                num_train_batches = self.x.shape[0] // self.batch_size
                train_batch_costs = [None] * num_train_batches
                train_batch_accs = [None] * num_train_batches

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                progbar = ProgressBar(max_value=num_train_batches)

                for b in range(num_train_batches):
                    batch = self.train_reader.next()

                    # Run forward and backward (backprop)
                    train_batch_costs[b], predictions, _, stateful_state = session.run(
                        [self.cost, self.pred, self.apply_placeholder_op, self.rnn_state],
                        feed_dict = {
                            self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                            self.state_placeholder : stateful_state
                        }
                    )
                    train_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                    progbar.update(b)
                progbar.finish()

                # Validation
                self.val_reader = read_data_generator(self.x_val, self.y_val, self.lengths_val, batch_size=self.batch_size)

                num_val_batches = self.x_val.shape[0] // self.batch_size
                val_batch_costs = [None] * num_val_batches
                val_batch_accs = [None] * num_val_batches

                stateful_state = np.zeros((2, self.batch_size, self.hidden_size), dtype=np.float32)  # 2 for c and h

                progbar = ProgressBar(max_value=num_val_batches)
                for b in range(num_val_batches):
                    batch = self.val_reader.next()

                    # Run forward, but not backprop
                    val_batch_costs[b], predictions, stateful_state = session.run(
                        [self.cost, self.pred, self.rnn_state],
                        feed_dict = {
                            self.x_batch : batch[0], self.y_batch : batch[1], self.w_batch : batch[2],
                            self.state_placeholder : stateful_state
                        }
                    )
                    val_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                    progbar.update(b)
                progbar.finish()

                # Validation accuracy is the mean accuracy over batch accuracies
                print(
                    'TRAIN (cost/acc): %.4f/%.2f%%, VAL (cost/acc): %.4f/%.2f%%' % (
                        np.mean(train_batch_costs), np.mean(train_batch_accs),
                        np.mean(val_batch_costs), np.mean(val_batch_accs)
                    )
                )

                if e % 2 == 0:
                    self.saver.save(session, 'simplelstm_model', global_step=e)

            # Testing
            self.te_reader = read_data_generator(self.x_te, self.y_te, self.lengths_te, batch_size=1)

            init_state = np.zeros((2, 1, self.hidden_size), dtype=np.float32)  # 2 for c and h

            num_te_batches = self.x_te.shape[0]
            te_batch_accs = [None] * num_te_batches

            progbar = ProgressBar(max_value=num_te_batches)
            for b in range(num_te_batches):
                batch = self.te_reader.next()

                # Run forward, but not backprop
                loss, predictions = session.run(
                    [self.loss, self.pred],
                    feed_dict={
                        self.x_batch: batch[0], self.y_batch: batch[1], self.w_batch: batch[2],
                        self.state_placeholder: init_state
                    }
                )
                te_batch_accs[b] = compute_framewise_accuracy(predictions, batch[1], batch[2])
                progbar.update(b)
            progbar.finish()

            print(
                'TRAIN (acc): %.2f%%, VAL (acc): %.2f%%, TE (acc): %.2f%%' %
                (np.mean(train_batch_accs), np.mean(val_batch_accs), np.mean(te_batch_accs))
            )

