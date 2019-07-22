# Deep Recurrent Attention Model (DRAM)
This repository contains code developed during my Summer 2019 internship at BICI. It is meant to be used
as a DRAM API which can be applied to various computer vision tasks. Functionality can and should be added to
adapt data for use with the network.

DRAM model developed based on research of Ba et al. (2015), Mnih et al. (2014), and Momeni et al. (2018).

utils.py, MNIST data extraction and training function style from github repository for: 

CS 20: "TensorFlow for Deep Learning Research" 

## Files and Usage

### utils.py
Currently contains basic functionality to safely create local directories, download files from online sources,
and convert tensor data to tf.data.Dataset objects for compatibility with DRAM model.

### datasets.py
Each class defines its own dataset. Within the class, preprocessing is done to convert data
and labels to a valid tuple, which can be passed into utils.convert_to_dataset() to obtain test and training datasets. 
Should return the training and test datasets.

Currently contains data for: MNIST

### BaseLayers.py
Defines basic layers used for convolutional and fully-connected components of the DRAM model under the Layers class. 

### config.py
Contains all parameters to be used during training and data storage. 

### NetworkLib.py
Defines classes for each componenet of the DRAM network. DRAM.py stores an instance of each network, and calls
with the appropriate parameters to obtain the desired output.

### DRAM.py
Main file which implements model construction and training. 

## To-Do:
- [x] Complete location network learning policy
- [ ] Develop more extensive and comprehensive metrics
- [ ] Track locations visited and create visual plot/animation of learning
