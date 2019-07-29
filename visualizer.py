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
    img_idx_range = config.verbose
    object_labels = config.object_labels
    
    # factors used to correct location and bounding box center
    hw = img_h / 2
    g_size = config.glimpse_size

    for img_idx in range(img_idx_range):
        utils.make_dir(config.image_dir_name + 'step{}/image{}'.format(step, img_idx))

        fig, ax = plt.subplots(1, 2)
        # map locations from [-1, 1] to [0, 28] image space
        locations[img_idx] = (locations[img_idx] * hw) + 14

        for glimpse in range(num_glimpses):
            glimpse_fig, glimpse_ax = plt.subplots(1)
            
            img = np.squeeze(images[img_idx])
            ax[0].imshow(img, cmap='Greys', interpolation='none')
            glimpse_ax.imshow(img, cmap='Greys', interpolation='none')

            location = locations[img_idx, glimpse]
            if (glimpse == 0):
                color = 'green'
            elif (glimpse == num_glimpses - 1):
                color = 'red'
            else:
                color = 'orange'

            glimpse_bbox = create_bbox((location[0] - g_size/2, location[1] - g_size/2), g_size, g_size, color=color)
            glimpse_ax.add_patch(glimpse_bbox)

            glimpse_png = config.image_dir_name + 'step{}/image{}/glimpse{}.png'.format(step, img_idx, glimpse)
            glimpse_fig.savefig(glimpse_png, bbox_inches='tight')
            plt.close(glimpse_fig)

            bbox = create_bbox((location[0] - g_size/2, location[1] - g_size/2), g_size, g_size, color=color)
            ax[0].add_patch(bbox)
        
        # Plot probability bar chart
        label_pos = np.arange(len(object_labels))
        preds_max_idx = np.argmax(preds[img_idx])
        prediction = preds[img_idx, preds_max_idx]
        prob = utils.truncate(prediction, 4)  
        label_max_idx = np.argmax(labels[img_idx])

        if preds_max_idx == label_max_idx:
            color = 'green'
        else:
            color = 'red'

        ax[1].set(adjustable='box')
        ax[1].bar(label_pos, preds[img_idx], align='center')
        ax[1].set_xticks(label_pos)
        ax[1].set_xticklabels(object_labels)
        ax[1].set_ybound(lower=0., upper=1.)
        ax[1].set_title('Prediction {} with prob {}, label {}'.format(preds_max_idx, prob, label_max_idx), color=color)

        png_name = config.image_dir_name + 'step{}/image{}/full_plot.png'.format(step, img_idx)
        fig.savefig(png_name, bbox_inches='tight')
        plt.close(fig)

def create_bbox(xy, width, height, color='green', linewidth=1.5, alpha=1):
    return patches.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha)

