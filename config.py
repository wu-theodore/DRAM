class Config(object):
    """
    Define hyperparameters here.
    """
    # Saver/Writer details
    checkpoint_path = 'checkpoints/mnist/'
    checkpoint_name = 'DRAM-mnist'
    report_step = 50
    graphs_path = 'graphs/mnist/'

    # Visualizer details
    visualize_step = 500
    image_dir_name = 'images/'

    # Training parameters
    lr = 0.01
    num_epochs = 20
    max_global_norm = 5.0
    isTraining = True
    isVisualize = True

    # Dataset details:
    batch_size = 100
    height = 28
    width = 28
    color_channels = 1
    num_classes = 10

    # Glimpse network and Context network parameters
    num_glimpses = 4
    glimpse_size = 8
    glimpse_scale = 2
    num_patches = 2
    first_conv_filters = 8
    kernel_size1 = 5
    kernel_size2 = 3
    kernel_size3 = 3
    strides = 1
    maxpool_window_size = 2
    maxpool_strides = 2
    feature_vector_size = 64

    # LSTM parameters
    cell_size = feature_vector_size

    # Location network parameters
    loc_net_dim = 2
    loc_net_stddev = 0.001

    # Classification network parameters
    classification_net_fc_dim = feature_vector_size / 4

    # Context network parameters
    coarse_size = 64

    # Loglikelihood parameters
    stddev = 0.25

    # Loss parameters
    reward_weight = 5