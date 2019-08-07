# Deep Recurrent Attention Model (DRAM)
This repository contains code for a Deep Recurrent Attention Model developed during my Summer 2019 internship at the Beijing Institute of Collaborative Innovation (BICI). As part of my research, I reproduced results from the original papers (Ba et al, Mnih et al) on the MNIST dataset and used Whole Slide Images (WSIs) from the Camelyon16 dataset to test the usability of DRAM on giga-pixel level images.

Implementation using TensorFlow 1.14.

## Inspiration
DRAM model developed based on research of Ba et al. (2015), Mnih et al. (2014), and Momeni et al. (2018).

utils.py, MNIST data extraction and training function style from github repository for: 

CS 20: "TensorFlow for Deep Learning Research" 

## Background 
DRAM is a deep learning network based off an RNN framework. This implementation primarily focuses on computer vision tasks. The model extracts glimpses from different locations of an input image and builds an internal representation before making a classification. Additional foveated glimpses can be concatenated to the original glimpse to introduce context to glimpses. A location and its corresponding glimpse are both encoded into feature vectors, which are multiplied element-wise to produce an input vector to the RNN network. LSTM cells were used in the RNN network for their stable learning dynamic. Each LSTM output is fed to a location network which determines the next glimpse location. The final LSTM state is used to classify the image. 

![Visualization of Model Architecture](https://github.com/theowu23451/DRAM/blob/master/media/DRAM%20Architecture.JPG)

**Source:** Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. “Recurrent Models of Visual At-tention”. In:CoRRabs/1406.6247 (2014). arXiv:1406.6247.url:http://arxiv.org/abs/1406.6247.

The location network is trained using the REINFORCE method, which emits a loglikelihood which tuhe optimizer maximizes. The classification network, RNN core network, and glimpse network are trained via backpropagation using the cross-entropy loss from classification. Additionally, a baseline is selected which regresses towards the expected value of the reward. In the image classification scenario, this reward is determined by the number of correct predictions the model makes. The baseline is used to reduce variance and is learned by minimizing the mean squared error between the reward signals and the baseline. A hybrid loss is constructed via summation of the mean baseline error, the cross-entropy loss, and the negation of the loglikelihood from the location network. The model uses the Adam optimizer to train this hybrid loss.

## Features
- Functional DRAM model designed with TensorFlow.
- Customizable config file allows modification of hyperparameters.
- File writer logs graph and summaries which can be displayed on Tensorboard.
- Saver writes checkpoints to save graph variables.
- Visualizer uses MatplotLib and ImageMagick to create possibilities plot and glimpse animation, respectively.

## Usage
Change parameters and data directories by directly editing config.py.

New datasets should be added to datasets.py as a new class, with a `get_dataset(self)` method which returns two tf.data.Dataset objects for training and testing data. Then, change the `self.dataset` variable in the `__init__` method of the DRAM class in DRAM.py to refer to the new dataset class.

To build model and initiate training/testing, run DRAM.py with python. Ensure TensorFlow version is up-to-date.
```
python .\DRAM.py
```
To view graph, run tensorboard with the appropriate logdir.
```
tensorboard --logdir GRAPH_DIR
```
Try adding flag `--host localhost` if tensorboard cannot read graph from system.

## Results
Obtained approximately 96% test accuracy on MNIST after 20 epochs of training. Used a learning rate of 0.01 and a glimpse size of 8. Accuracy can likely be marginally improved through fine-tuning the dimensions of the feature vectors and of the convolutional layer filters. 

### Epoch 1:
![Sample image and probability plot](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch0_full_plot.png) ![Glimpse animation](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch0_glimpse_anim.gif)

### Epoch 10:
![Sample image and probability plot](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch10_full_plot.png) ![Glimpse animation](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch10_glimpse_anim.gif)

### Epoch 20:
![Sample image and probability plot](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch20_full_plot.png) ![Glimpse animation](https://github.com/theowu23451/DRAM/blob/master/media/img0_epoch20_glimpse_anim.gif)

### Tensorboard Summaries:
![Scalar summary graphs](https://github.com/theowu23451/DRAM/blob/master/media/Scalar%20Summaries.JPG)

## License
This project is licensed under the MIT License.
