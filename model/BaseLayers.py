import tensorflow as tf

class Layers(object):
    """
    A class containing functionality to customize 
    and create standard convolutional network layers.
    """
    def __init__(self):
        pass

    def conv_layer(self, inputs, filters, kernel_size, strides, padding='SAME', name='conv'):
        input_channels = inputs.shape[-1]
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            kernel = tf.compat.v1.get_variable('kernel', 
                                     shape=[kernel_size, kernel_size, input_channels, filters],
                                     initializer=tf.truncated_normal_initializer(), dtype=tf.float32)

            bias = tf.compat.v1.get_variable('bias', shape=[filters],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)
                                
            conv = tf.nn.conv2d(inputs, filter=kernel,
                                strides=[1, strides, strides, 1],
                                padding=padding, name='conv')

            out = tf.nn.relu(conv + bias)
            return out

    def max_pool(self, inputs, ksize, strides, padding='VALID', name='maxpool'):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            pool = tf.nn.max_pool2d(inputs, ksize=[1, ksize, ksize, 1],
                                  strides=[1, strides, strides, 1], 
                                  padding=padding, name=name)
            return pool

    def fully_connected(self, inputs, out_dim, name):
        in_dim = inputs.shape[-1]
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable("weights", shape=[in_dim, out_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            b = tf.compat.v1.get_variable("biases", shape=[out_dim],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)

            out = tf.matmul(inputs, w) + b
            return out
