import tensorflow as tf
import numpy as np

x = tf.get_variable('var_x', dtype=tf.float32, initializer=tf.random_uniform((1, 1), -0.1, 0.1))
z = tf.get_variable('var_z', dtype=tf.float32, initializer=tf.zeros((1, 1)))
prod = tf.multiply(x,z)
init_op = tf.global_variables_initializer()

# A = tf.Graph()
# with A .as_default() as gA:
#     x = tf.Variable(tf.random_uniform((1,1), -0.1, 0.1), dtype=tf.float32)
#     init_op = tf.global_variables_initializer()
# tf.reset_default_graph()
#
# B = tf.Graph()
# with B.as_default() as gB:
#     # y = tf.get_variable('x', shape=[1,1])
#     init_op = tf.global_variables_initializer()
#     # y = tf.multiply(x, 2)
# tf.reset_default_graph()


if __name__ == '__main__':
    with tf.Session() as session:
        session.run([init_op])
        ret = session.run(prod)
        print ret


