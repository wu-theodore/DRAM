import tensorflow as tf
import numpy as np
from BaseLayers import Layers


class GlimpseNet(object):

    def __init__(self, input_img, config):
        self.layers = Layers()
        self.config = config

        # Dataset inputs
        self.input_img = input_img
        self.batch_size = config.batch_size

    def extract_glimpse(self, location, glimpse_size):
        location = tf.stop_gradient(location)
        with tf.name_scope("glimpse_sensor"):
            glimpse = tf.image.extract_glimpse(self.input_img, 
                                     [glimpse_size, glimpse_size],
                                     location, centered=True, normalized=True,
                                     name='extract_glimpse')
            return glimpse

    def __call__(self, location):
        """
        When called with a location, returns the feature vector associated with 
        glimpses extracted at that location. 
        
        If num_patches > 1, foveate input glimpse by glimpse_scale
        by extracting additional larger patches, downsampling, and concatenating.
        """
        with tf.compat.v1.variable_scope("glimpse_network", reuse=tf.compat.v1.AUTO_REUSE):
            input_glimpse = tf.Variable(np.empty([self.batch_size,
                                        self.config.glimpse_size, self.config.glimpse_size,
                                        0]), dtype=tf.float32, trainable=False, name="accum")
            for patch in range(self.config.num_patches):
                patch_glimpse = self.extract_glimpse(location,
                                                    self.config.glimpse_size * (self.config.glimpse_scale ** patch))

                patch_glimpse = tf.image.resize(patch_glimpse, 
                                                       [self.config.glimpse_size, self.config.glimpse_size])

                input_glimpse = tf.concat([input_glimpse, patch_glimpse], -1, name="concatenate")

            # Calculate image feature vector.
            with tf.compat.v1.variable_scope("feature_network", reuse=tf.compat.v1.AUTO_REUSE):
                with tf.compat.v1.variable_scope("convolutions", reuse=tf.compat.v1.AUTO_REUSE):
                    conv1 = self.layers.conv_layer(input_glimpse, filters=self.config.conv1_filters, 
                                                   kernel_size=self.config.kernel_size1, strides=self.config.strides,
                                                   padding='SAME', name='conv1')

                    pool = self.layers.max_pool(conv1, self.config.maxpool_window_size, self.config.maxpool_strides,
                                                padding='VALID', name='maxpool')

                    conv2 = self.layers.conv_layer(pool, filters=self.config.conv_2_filters, 
                                                   kernel_size=self.config.kernel_size2, strides=self.config.strides,
                                                   padding='SAME', name='conv2')
                    
                    conv2 = tf.layers.batch_normalization(conv2, training=True)

                    conv3 = self.layers.conv_layer(conv2, filters=self.config.conv_3_filters, 
                                                   kernel_size=self.config.kernel_size3, strides=self.config.strides,
                                                   padding='SAME', name='conv3')

                with tf.compat.v1.variable_scope('fully_connected', reuse=tf.compat.v1.AUTO_REUSE):
                    flattened_dim = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
                    flattened = tf.reshape(conv3, [-1, flattened_dim], name='flatten') 
                    fc = self.layers.fully_connected(flattened, self.config.feature_vector_size, 'fc_image')
                    fc = tf.layers.batch_normalization(fc, trainable=True)
                    image_vector = tf.compat.v1.nn.relu(fc) 
            
            # Calculate location vector
            with tf.compat.v1.variable_scope("location_network", reuse=tf.compat.v1.AUTO_REUSE):
                fc = self.layers.fully_connected(location, self.config.feature_vector_size, name='fc_location')
                location_vector = tf.compat.v1.nn.relu(fc)

            with tf.compat.v1.variable_scope("element_multiplication"):
                feature_vector = tf.math.multiply(location_vector, image_vector, name='element_multiplication')

            return feature_vector

class LocationNet(object):
    def __init__(self, config):
        self.loc_net_dim = config.loc_net_dim
        self.location_stddev = config.loc_net_stddev
        self.layers = Layers()
    
    def __call__(self, vector):
        """
        Call with vector to obtain next location.

        Next location (loc) is a 2-D tensor which represents (x, y)

        Use vector r2 from RNN network. 
        """
        with tf.compat.v1.variable_scope("emission_network", reuse=tf.compat.v1.AUTO_REUSE):
            mean = tf.stop_gradient(tf.clip_by_value(self.layers.fully_connected(vector, self.loc_net_dim, "loc_fc"), -1, 1))
            loc = mean + tf.random.normal((tf.shape(vector)[0], 2), stddev= self.location_stddev)
            loc = tf.stop_gradient(loc)
        return loc, mean

class RecurrentNet(object):
    def __init__(self, config, location_network, glimpse_network, classification_network):
        self.batch_size = config.batch_size
        self.location_network = location_network
        self.glimpse_network = glimpse_network
        self.class_net = classification_network

    def __call__(self, inputs, cell):
        """
        Takes list of length num_glimpses, with first element being initial glimpse.
        
        Dynamically updates list with glimpse vectors from locations determined by location network.

        Glimpses are passed through two layer RNN built using an LSTM cell.
        """
        # Set initial values
        loc_array = []
        mean_loc_array = []
        outputs = []
        inputs = inputs
        state = cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        prev = None
        
        with tf.compat.v1.name_scope("core_network"):
            for iteration, input_vector in enumerate(inputs):
                if prev is not None:
                    loc, mean = self.location_network(prev)
                    input_vector = self.glimpse_network(loc)

                    # Add new location to output array of locations.
                    loc_array.append(loc)
                    mean_loc_array.append(mean)

                with tf.compat.v1.variable_scope("rnn_network", reuse=tf.compat.v1.AUTO_REUSE):
                    output, state = cell(input_vector, state)
                    outputs.append(output)

                prev = output

        final_feature_vector = outputs[-1]
        logits = self.class_net(final_feature_vector)

        return logits, outputs, loc_array, mean_loc_array


class ContextNet(object):
    def __init__(self, config):
        self.layers = Layers()
        self.config = config
        self.batch_size = config.batch_size

    def __call__(self, input_img):
        """
        Extracts a coarse_image of the input image.

        Used to calculate a coarse image vector to be used in determining
        initial state of the second LSTM layer.
        """
        with tf.compat.v1.variable_scope("context_network", reuse=tf.compat.v1.AUTO_REUSE):
            self.coarse_image = tf.image.resize_images(input_img, 
                                                       [self.config.coarse_size, self.config.coarse_size], name='resize')
                                                       
            conv1 = self.layers.conv_layer(self.coarse_image, filters=self.config.first_conv_filters, kernel_size=self.config.kernel_size1, 
                                           strides=self.config.strides, padding='SAME', name='conv1')
            conv2 = self.layers.conv_layer(conv1, filters=self.config.first_conv_filters * 2, kernel_size=5,
                                           strides=1, padding='SAME', name='conv2')
            conv3 = self.layers.conv_layer(conv2, filters=self.config.first_conv_filters * 4, kernel_size=5,
                                           strides=1, padding='SAME', name='conv3')
            flattened_dim = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
            initial_vector = tf.reshape(conv3, [-1, flattened_dim], name='flatten')
            return initial_vector

class ClassificationNet(object):
    def __init__(self, config):
        self.layers = Layers()
        self.config = config

    def __call__(self, feature_vector):
        """
        Outputs logits of input feature vector.
        """
        with tf.compat.v1.variable_scope("classification_network", reuse=tf.compat.v1.AUTO_REUSE):
            fc = self.layers.fully_connected(feature_vector, self.config.classification_net_fc_dim, name='class_fc1')
            fc = tf.compat.v1.nn.relu(fc)
            logits = self.layers.fully_connected(fc, self.config.num_classes, name='class_fc2')
        return logits

class BaselineNet(object):
    def __init__(self, config):
        self.layers = Layers()
        self.config = config
    
    def __call__(self, feature_vector):
        """
        Outputs baseline value of input feature vector.
        """
        with tf.compat.v1.variable_scope("baseline", reuse=tf.compat.v1.AUTO_REUSE):
            baseline = self.layers.fully_connected(feature_vector, 1, name='baseline_fc')
            baseline = tf.compat.v1.nn.relu(baseline)

        return baseline
