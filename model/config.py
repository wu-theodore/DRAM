class Config(object):
    """
    Define hyperparameters here.
    """
    # Training parameters
    lr = 0.01
    num_epochs = 5
    max_global_norm = 5.0
    isTraining = True
    

    # Dataset details:
    batch_size = 100
    height = 28
    width = 28
    color_channels = 1
    object_labels = ('0', '1', '2', '3', '4',
                     '5', '6', '7', '8', '9')
    num_classes = len(object_labels)

    # Glimpse extraction parameters
    num_glimpses = 5
    glimpse_size = 8
    glimpse_scale = 2
    num_patches = 2

    # Glimpse network parameters
    conv1_filters = 16
    conv_2_filters = 32
    conv_3_filters = 64
    kernel_size1 = 7
    kernel_size2 = 3
    kernel_size3 = 3
    strides = 2
    maxpool_window_size = 3
    maxpool_strides = 2
    feature_vector_size = 64

    # LSTM parameters
    cell_size = feature_vector_size

    # Location network parameters
    loc_net_dim = 2
    loc_net_stddev = 0.001

    # Classification network parameters
    classification_net_fc_dim = 32

    # Context network parameters
    coarse_size = 64

    # Loglikelihood parameters
    stddev = 0.25

    # Loss parameters
    reward_weight = 5

    # Dataset details
    data_path = 'data/mnist/'

    # Saver/Writer details
    checkpoint_path = 'checkpoints/mnist/lr={}n_glimpse={}glimpse_size{}n_patch={}/'.format(lr, num_glimpses, glimpse_size, num_patches)
    checkpoint_name = 'DRAM-mnist'
    report_step = 50
    graphs_path = 'graphs/mnist/lr={}n_glimpse={}glimpse_size{}n_patch={}/'.format(lr, num_glimpses, glimpse_size, num_patches)

   # Visualizer details
    isVisualize = True
    isAnimate = True
    visualize_step = 550
    verbose = 5 # Adjusts how many images to display for plot
    image_dir_name = 'images/mnist/lr={}n_glimpse={}glimpse_size{}n_patch={}/'.format(lr, num_glimpses, glimpse_size, num_patches)
