import os
import posixpath
import utils
import struct

import numpy as np
import tensorflow as tf

class MNIST(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def download_mnist(self, path):

        utils.make_dir(path)

        # file download information
        url = 'http://yann.lecun.com/exdb/mnist'

        filenames = ["train-images-idx3-ubyte.gz",   
                    "train-labels-idx1-ubyte.gz",
                    "t10k-images-idx3-ubyte.gz",
                    "t10k-labels-idx1-ubyte.gz"]

        expected_bytes = [9912422, 28881, 1648877, 4542]

        for filename, byte in zip(filenames, expected_bytes):
            download_url = posixpath.join(url, filename)
            local_path = os.path.join(path, filename)
            utils.download_file(download_url, local_path, byte, unzip=True)


    def create_data(self, path, dataset, flatten=True):
        if dataset != 'train' and dataset != 't10k':
            raise NameError("not a usable dataset: use 'train' or 't10k'")
        if not os.path.exists(path):
            raise NameError("nonexistent file.")

        label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
        with open(label_file, 'rb') as file:
            _, length = struct.unpack(">II", file.read(8))
            labels = np.fromfile(file, dtype=np.int8)

            """For each image, correct labels to 1 and all others to 0"""
            new_labels = np.zeros((length, 10))
            new_labels[np.arange(length), labels] = 1

        img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
        with open(img_file, 'rb') as file:
            _, length, rows, cols = struct.unpack(">IIII", file.read(16))
            imgs = np.fromfile(file, dtype=np.uint8)
            imgs.reshape(length, rows, cols)
            imgs = imgs.astype(np.float32) / 255.0
            if flatten:
                imgs = imgs.reshape([length, -1])

        return (imgs, new_labels)

    def read_mnist(self, path, flatten=True, num_training=55000):
        """ Download file, then read file information.
            return tuples of numpy arrays in format 
            (imgs, labels).

            Split training set based on num_training, leaving rest as validation set.
        """
        self.download_mnist(path)

        # Process training data
        (imgs, labels) = self.create_data(path, 'train', flatten)

        #Shuffle training set.
        indices = np.random.permutation(labels.shape[0])
        train_idx, val_idx = indices[:num_training], indices[num_training:]

        train_set, train_labels = imgs[train_idx, :], labels[train_idx, :]
        val_set, val_labels = imgs[val_idx, :], labels[val_idx, :]

        # Process test data
        test = self.create_data(path, 't10k', flatten)
        return (train_set, train_labels), (val_set, val_labels), test

    def get_mnist_dataset(self):
        """ Given batch_size, returns 
            Dataset object from mnist data
            with batch sizes of input batch_size
        """

        mnist_folder = 'data/mnist/'

        train, val, test = self.read_mnist(mnist_folder)

        # Create tf Datasets for each.
        train_data = utils.convert_to_dataset(train, self.batch_size)

        test_data = utils.convert_to_dataset(test, self.batch_size)

        return train_data, test_data


