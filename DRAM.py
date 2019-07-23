import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_probability as tfp
import time

import utils
import datasets
from NetworkLib import GlimpseNet, LocationNet, ContextNet, ClassificationNet, RecurrentNet, BaselineNet

import config

class DRAM(object):
    def __init__(self):
        self.config = config.Config()
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                 trainable=False, name='global_step')
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.isTraining = self.config.isTraining

        self.dataset = datasets.MNIST(self.batch_size)

    def get_data(self):
        """
        Get dataset and create iterators for test and train.
        """
        with tf.name_scope('data'):
            train_dataset, test_dataset = self.dataset.get_mnist_dataset()
            iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_dataset),
                                                                 tf.compat.v1.data.get_output_shapes(train_dataset))
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
        self.gn = GlimpseNet(self.img, self.config)

        self.ln = LocationNet(self.config)

        self.class_net = ClassificationNet(self.config)
        
        self.context_net = ContextNet(self.config)

        self.rnn = RecurrentNet(self.config, self.ln, self.gn, self.class_net)

        self.baseline_net = BaselineNet()

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
        self.mean_loc_array = [self.init_location]

        self.init_glimpse = [self.gn(self.init_location)]
        self.init_glimpse.extend([0] * (self.config.num_glimpses - 1))

        self.logits, self.outputs, self.states, locations, mean_locations = self.rnn(self.init_glimpse, self.state_init_input, 
                                                                                     self.LSTM_cell1, self.LSTM_cell2)
        self.loc_array += locations
        self.mean_loc_array += mean_locations

        self.sampled_locations = tf.concat(self.loc_array, axis=0)
        self.mean_locations = tf.concat(self.mean_loc_array, axis=0)
        self.sampled_locations = tf.reshape(self.sampled_locations, (self.config.num_glimpses, self.batch_size, 2))
        self.sampled_locations = tf.transpose(self.sampled_locations, [1, 0, 2])
        self.mean_locations = tf.reshape(self.mean_locations, (self.config.num_glimpses, self.batch_size, 2))
        self.mean_locations = tf.transpose(self.mean_locations, [1, 0, 2])
        prefix = tf.expand_dims(self.init_location, 1)
        self.sampled_locations = tf.concat([prefix, self.sampled_locations], axis=1)
        self.mean_locations = tf.concat([prefix, self.mean_locations], axis=1)

    def loglikelihood(self):
        with tf.name_scope("loglikelihood"):
            stddev = self.config.stddev
            mean = tf.stack(self.mean_loc_array)
            sampled = tf.stack(self.loc_array)
            gaussian = tfp.distributions.Normal(mean, stddev)
            logll = gaussian.log_prob(sampled)
            logll = tf.reduce_sum(logll, 2)
            logll = tf.transpose(logll)  
        return logll

    def loss(self):
        with tf.name_scope("loss"):
            # Cross entropy
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.cross_ent = tf.reduce_mean(entropy, name='cross_ent')

            # Baseline MSE
            self.baseline = self.baseline_net(self.states[1][0])
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.rewards = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            self.baseline_mse = tf.reduce_mean(tf.square(self.rewards - self.baseline), name='baseline_mse')

            # Loglikelihood
            self.logll = self.loglikelihood()
            self.baseline_term = self.rewards - tf.stop_gradient(self.baseline)
            self.logllratio = tf.reduce_mean(self.logll * self.baseline_term, name='loglikelihood_ratio')

            # Total Loss
            self.hybrid_loss = -self.logllratio + self.cross_ent + self.baseline_mse

    def optimize(self):
        """
        Optimize action network using backpropagation by minimizing loss.
        """
        self.opt = tf.compat.v1.train.AdamOptimizer(self.config.lr).minimize(self.hybrid_loss, 
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
            tf.compat.v1.summary.scalar('cross_entropy', self.cross_ent)
            tf.compat.v1.summary.scalar('baseline_mse', self.baseline_mse)
            tf.compat.v1.summary.scalar('loglikelihood', self.logllratio)
            tf.compat.v1.summary.scalar('hybrid_loss', self.hybrid_loss)
            tf.compat.v1.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.compat.v1.summary.merge_all()

    def train(self, num_epochs, isTraining):
        """
        1) Make checkpoint folder (using config.checkpoint_path) and saver.
        2) Alternate between training and testing for each epoch.
        """
        self.num_epochs = num_epochs
        utils.make_dir(self.config.checkpoint_path)
        writer = tf.compat.v1.summary.FileWriter(self.config.graphs_path, tf.compat.v1.get_default_graph())

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.config.checkpoint_path))
            if ckpt and ckpt.model_checkpoint_path:
                print("Checkpoint path found, restoring:")
                saver.restore(sess, ckpt.model_checkpoint_path)
            step = self.gstep.eval()

            if isTraining:
                for epoch in range(num_epochs):
                    step = self.train_one_epoch(sess, saver, self.train_init, writer, step, epoch)
                    self.eval_once(sess, self.test_init, writer, step, epoch)
            else:
                self.eval_once(sess, self.test_init, writer, step, epoch)

        writer.close()

    def train_one_epoch(self, sess, saver, init, writer, step, epoch):
        print("Training epoch {0}/{1}".format(epoch, self.num_epochs))
        sess.run(init)
        start = time.time()
        num_batches = 0
        total_loss = 0
        try:
            while True:
                cross_ent, hybrid_loss, logllratio, base_mse, accuracy, _, summary = sess.run([self.cross_ent, self.hybrid_loss, 
                                                                                               self.logllratio, self.baseline_mse,
                                                                                               self.accuracy, self.opt, self.summary_op])
                writer.add_summary(summary, global_step=step)
                if (step + 1) % self.config.report_step == 0:
                    print("----------------LOSSES----------------")
                    print("Accuracy at step {0}: {1}".format(step, accuracy))
                    print("Cross entropy loss at step {0}: {1}".format(step, cross_ent))
                    print("Baseline MSE at step {0}: {1}".format(step, base_mse))
                    print("Loglikelihood ratio at step {0}: {1}".format(step, logllratio))
                    print("Total loss at step {0}: {1}".format(step, hybrid_loss))
                    print("--------------------------------------\n")
                num_batches += 1
                total_loss += hybrid_loss
                step += 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, self.config.checkpoint_path + self.config.checkpoint_name, global_step=self.gstep)
        print("Average loss per batch: {0}".format(total_loss / num_batches))
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
        print("-----------------EVAL----------------")
        print("Average accuracy: {}".format(total_acc / num_batches))
        print("Time taken: {}".format(time.time() - start))
        print('\n')

if __name__ == "__main__":
    test = DRAM()
    test.build()
    test.train(test.num_epochs, test.isTraining)

            

