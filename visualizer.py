import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import utils

def plot_glimpse(config, images, locations, preds, labels, step):
    """
    For each image in images, draws bounding boxes 
    corresponding to glimpse locations.

    First glimpse is colored green, intermediate 
    glimpses are orange, and terminal glimpse is red.

    Saves image with overlaid bounding boxes to a local dir.
    """
    batch_size, img_h, img_w, channels = images.shape
    num_glimpses = config.num_glimpses
    
    # factors used to correct location and bounding box center
    hw = img_h / 2
    g_size = config.glimpse_size

    for img_idx in range(int(batch_size/4)):
        utils.make_dir(config.image_dir_name + 'step{}/'.format(step))

        fig, ax = plt.subplots(1)
        # map locations from [-1, 1] to [0, 28] image space
        locations[img_idx] = (locations[img_idx] * hw) + 14

        for glimpse in range(num_glimpses):
            img = np.squeeze(images[img_idx])
            ax.imshow(img, cmap='Greys', interpolation='none')
            
            location = locations[img_idx, glimpse]
            if (glimpse == 0):
                bbox = create_bbox((location[0] - g_size/2, location[1] - g_size/2), g_size, g_size, color='green')

            elif (glimpse == num_glimpses - 1):
                bbox = create_bbox((location[0] - g_size/2, location[1] - g_size/2), g_size, g_size, color='red')

            else:
                bbox = create_bbox((location[0] - g_size/2, location[1] - g_size/2), g_size, g_size, color='orange')

            ax.add_patch(bbox)

        png_name = config.image_dir_name + 'step{}/image{}.png'.format(step, img_idx)
        fig.savefig(png_name, bbox_inches='tight')
        plt.close(fig)

def create_bbox(xy, width, height, color='green', linewidth=1.5, alpha=1):
    return patches.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha)

