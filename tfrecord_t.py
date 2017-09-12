import tensorflow as tf
import numpy as np
import random
import time

FEA_DIM = 224*224*3

class Data(object):
    def __init__(self):
        self.vectors = []
        self.labels = []
        self.L = [740,5508,120] #,667,968,444,5420,591,1860,3435,941,704,310,968,3141,3694,3874,2386,1860,979,1401,1105,461,1261,3313,3094,3017,5448,1850]
        for i in range(len(self.L)):
            print i
            length = self.L[i]

            self.vectors.append(np.random.randint(0,10,(length,FEA_DIM)).astype(dtype=np.float32))
            self.labels.append(np.random.randint(0,10,(length,)).astype(dtype=np.int64))

data = Data()

print 'Writing'
st_time = time.time()
writer = tf.python_io.TFRecordWriter("prova.tfrecords")
for i in range(len(data.L)):
    inputs = data.vectors[i]
    labels = data.labels[i]

    example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list = {
                'aaaa': tf.train.FeatureList(
                    feature=[tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in inputs]
                )
            }
        )
    )

    # example = tf.train.Example(
    #     features = tf.train.Features(
    #         feature = {
    #             'labels' : tf.train.Feature(
    #                 int64_list=tf.train.Int64List(value=labels))
    #         }
    #     )
    # )
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()

print time.time()-st_time

# record_iterator = tf.python_io.tf_record_iterator(path="prova.tfrecords")
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#
#     labels = example.features.feature['labels'].int64_list.value
#     print 1


file_queue = tf.train.string_input_producer(["prova.tfrecords"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)

# features = tf.parse_single_example(
#     serialized_example,
#     features={
#         # We know the length of both fields. If not the
#         # tf.VarLenFeature could be used
#         'labels': tf.FixedLenSequenceFeature([5], tf.float32),
#     })
#
# return labels

# sequence_features =

_, sequence_features = tf.parse_single_sequence_example(
    serialized_example, sequence_features={
        "aaaa": tf.FixedLenSequenceFeature([FEA_DIM], dtype=tf.float32)
    }
)
aaaa = sequence_features['aaaa']

# input_tensors = sequence['inputs']


# labels = read_and_decode_single_example("prova.tfrecords")

sess = tf.Session()

# Required. See below for explanation
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# grab examples back.
# first example from file
# print sess.run([aaaa])
print 'Reading'
st_time = time.time()
for i in range(len(data.L)):
    sess.run([aaaa])
print time.time() - st_time
# # second example from file
# print sess.run([aaaa])