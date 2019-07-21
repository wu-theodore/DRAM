import tensorflow as tf

class Layers(object):
    def __init__(self):
        pass

    def conv_layer(self, inputs, filters, kernel_size, strides, padding='SAME', name='conv'):
        """
        Basic convolutional layer.
        """
        input_channels = inputs.shape[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable('kernel', 
                                     shape=[kernel_size, kernel_size, input_channels, filters],
                                     initializer=tf.truncated_normal_initializer(dtype=tf.float32))

            bias = tf.get_variable('bias', shape=[filters],
                                initializer=tf.random_normal_initializer(dtype=tf.float32))
                                
            conv = tf.nn.conv2d(inputs, filter=kernel,
                                strides=[1, strides, strides, 1],
                                padding=padding, name='conv')

            out = tf.nn.relu(conv + bias, name=name)
            return out

    def max_pool(self, inputs, ksize, strides, padding='VALID', name='maxpool'):
        """
        A basic max pooling layer.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1],
                                  strides=[1, strides, strides, 1], 
                                  padding=padding, name=name)
            return pool

    def fully_connected(self, inputs, out_dim, name):
        """
        Basic fully connected layer.
        
        Output not ReLU'd
        """
        in_dim = inputs.shape[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w = tf.get_variable("weights", shape=[in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer(dtype=tf.float32))

            b = tf.get_variable("biases", shape=[out_dim],
                                initializer=tf.random_normal_initializer(dtype=tf.float32))

            out = tf.matmul(inputs, w) + b
            return tf.nn.relu(out)
