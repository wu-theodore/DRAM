import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import time

import utils
import datasets
from NetworkLib import ConvGlimpseNet, LocationNet, ContextNet, ClassificationNet, RecurrentNet

import config

class DRAM(object):
    def __init__(self):
        self.config = config.Config()
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                 trainable=False, name='global_step')
        self.num_epochs = self.config.num_epochs
        self.dataset = datasets.MNIST(self.config.batch_size)

    def get_data(self):
        """
        Get dataset and create iterators for test and train.
        """
        with tf.name_scope('data'):
            train_dataset, test_dataset = self.dataset.get_mnist_dataset()
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                    train_dataset.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, [-1, self.config.height, self.config.width, self.config.color_channels])
            self.train_init = iterator.make_initializer(train_dataset)
            self.test_init = iterator.make_initializer(test_dataset)

    def model_init(self):
        """
        Creates instances of each network from NetworkLib.py

        Defines LSTM cells for use in RNN
        """
        # Initiate Networks
        self.gn = ConvGlimpseNet(self.img, self.config)

        self.ln = LocationNet(self.config)

        self.class_net = ClassificationNet(self.config)
        
        self.context_net = ContextNet(self.config)

        self.rnn = RecurrentNet(self.config, self.ln, self.gn, self.class_net)

        # Initiate LSTM cells for RNN
        self.LSTM_cell1 = tf.keras.layers.LSTMCell(self.config.cell_size, 
                                                  activation=tf.nn.tanh, unit_forget_bias=True)
        self.LSTM_cell2 = tf.keras.layers.LSTMCell(self.config.cell_size, 
                                                  activation=tf.nn.tanh, unit_forget_bias=True)

    def inference(self):
        """
        1)  Creates initial state through context network.
        2)  Extracts first glimpse through glimpse network.
        3)  First glimpse is passed through RNN network, 
            which integrates RNN components, location/emission 
            network, and glimpse network.
            
            Returns final softmax output,
            a list of outputs from each layer,
            tuple of final RNN states, and 
            list of locations visited.

        4)  Final output is passed into classification network,
            where prediction is obtained.
        """

        self.state_init_input = self.context_net(self.img)

        self.init_location = tf.zeros([tf.shape(self.img)[0], 2], dtype=tf.float32, name='init_loc')
        self.loc_array = [self.init_location]

        self.init_glimpse = [self.gn(self.init_location)]
        self.init_glimpse.extend([0] * (self.config.num_glimpses - 1))

        # -------------------------------CHECK WHETHER LIST OF OUTPUTS IS NEEDED OR NOT-----------------------------------
        self.logits, self.outputs, self.states, locations = self.rnn(self.init_glimpse, self.state_init_input, 
                                                                             self.LSTM_cell1, self.LSTM_cell2)
        self.loc_array.append(locations)

    def loss(self):
        """
        Calculate cross entropy loss for classification.
        """
        with tf.name_scope("loss"):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.class_loss = tf.reduce_mean(entropy, name='loss')
    
    def optimize(self):
        """
        Optimize action network using backpropagation by minimizing loss.
        """
        self.opt = tf.train.AdamOptimizer(self.config.lr).minimize(self.class_loss, 
                                                global_step=self.gstep)

    def build(self):
        """
        Construct the Tensorflow session graph.
        """
        self.get_data()
        self.model_init()
        self.inference()
        self.loss()
        self.optimize()
        # ------------------------ STILL NEED LOG POLICY GRADIENT CODE ------------------------------- #
        self.eval()
        self.summaries()

    def eval(self):
        """
        Count number of correct predictions per batch.
        """
        with tf.name_scope("predict"):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    
    def summaries(self):
        """
        Create summaries to write to tensorboard.
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar('classification_loss', self.class_loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def train(self, num_epochs):
        """
        1) Make checkpoint folder (using config.checkpoint_path) and saver.
        2) Alternate between training and testing for each epoch.
        """
        self.num_epochs = num_epochs
        utils.make_dir(self.config.checkpoint_path)
        writer = tf.summary.FileWriter(self.config.graphs_path, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.config.checkpoint_path))
            if ckpt and ckpt.model_checkpoint_path:
                print("Checkpoint path found, restoring:")
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(num_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, step, epoch)
                self.eval_once(sess, self.test_init, writer, step, epoch)

        writer.close()

    def train_one_epoch(self, sess, saver, init, writer, step, epoch):
        print("Training epoch {0}/{1}".format(epoch, self.num_epochs))
        sess.run(init)
        start = time.time()
        num_batches = 0
        total_class_loss = 0
        try:
            while True:
                class_loss, _, summary = sess.run([self.class_loss, self.opt, self.summary_op])
                writer.add_summary(summary, global_step=step)
                if (step + 1) % self.config.report_step == 0:
                    print("Classification loss at step {0}: {1}".format(step, class_loss))
                num_batches += 1
                total_class_loss += class_loss
                step += 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, self.config.checkpoint_name, global_step=self.gstep)
        print("Average classification loss per batch: {0}".format(total_class_loss / num_batches))
        print("Time taken: {}".format(time.time() - start))
        return step
    
    def eval_once(self, sess, init, writer, step, epoch):
        sess.run(init)
        start = time.time()
        num_batches = 0
        total_acc = 0
        try:
            while True:
                acc, summary = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summary, global_step=step)
                total_acc += acc
                num_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print("Average accuracy: {}".format(total_acc / num_batches))
        print("Time taken: {}".format(time.time() - start))

if __name__ == "__main__":
    test = DRAM()
    test.build()
    test.train(test.num_epochs)

            

