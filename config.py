class Config(object):
    def __init__(self):
        """
        Define hyperparameters here.
        """
        # Saver/Writer details
        self.checkpoint_path = 'checkpoints/mnist/'
        self.checkpoint_name = 'DRAM-mnist'
        self.report_step = 50
        self.graphs_path = 'graphs/mnist/'

        # Learning parameters
        self.lr = 0.001
        self.num_epochs = 2
    
        # Dataset details:
        self.batch_size = 100
        self.height = 28
        self.width = 28
        self.color_channels = 1
        self.num_classes = 10

        # Glimpse network parameters
        self.num_glimpses = 16
        self.glimpse_size = 8
        self.glimpse_scale = 2
        self.num_patches = 2
        self.first_conv_filters = 16
        self.kernel_size = 5
        self.strides = 1
        self.feature_vector_size = 10

        # LSTM parameters
        self.cell_size = 256

        # Location network parameters
        self.loc_net_dim = 2

