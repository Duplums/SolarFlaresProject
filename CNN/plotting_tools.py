''' This claim provides some functions for DNN visualization and image visualization in general.'''

import matplotlib
matplotlib.use('Agg') # no display 'ON'
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Plotting_Tools:
    # We assume that a kernel has the following dim: h x w x depth x nb_filters 
    # In this plot, each row corresponds to a flatten filter of the kernel. 
    # The depth is reduced to 'max_depth'.

    @staticmethod
    def plot_kernel(kernel, max_depth = 20):
        assert type(kernel) == np.array and len(kernel.shape) == 4
        (h, w, d, n) = kernel.shape
        plt.figure(1, figsize=(min(d,max_depth), n))
        index = 1
        nrows = n
        ncols = min(max_depth, d)
        # Iterates through filters
        for k in range(n):
            # Iterates through the same filter and plot it on a single row
            for i in range(min(d, max_depth)):
                plt.subplot(nrows, ncols, index)
                plt.axis('off')
                plt.imshow(kernel[:,:,i, k], cmap='gray')
                index +=1 
        plt.show()
    
    
    # We assume that the pictures are stored in an array of shape h x w x c
    # (channel last) or c x h x w (channel first).
    @staticmethod
    def plot_pictures(pics, nrows, ncols, channel_last = True, labels = None, figsize=None,
                     save_fig=False, name_fig="pic.png"):
        assert type(pics) == np.ndarray and len(pics.shape) == 3
        if(channel_last):
            assert labels is None or len(labels) >= min(nrows*ncols, pics.shape[-1])
            (h, w, c) = pics.shape
        else:
            assert labels is None or len(labels) >= min(nrows*ncols, pics.shape[0])
            (c, h, w) = pics.shape
        if(figsize is None):
            figsize = (1.5*ncols, 1.5*nrows)
        plt.figure(figsize=figsize)
        if(labels is None):
            plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        for k in range(min(c, nrows*ncols)):
            plt.subplot(nrows, ncols, k+1)
            if(channel_last):
                plt.imshow(pics[:,:,k], cmap='gray')
            else:
                plt.imshow(pics[k,:,:], cmap='gray')
            if(labels is not None):
                plt.title(labels[k])
            plt.axis('off')
        
        if(save_fig):
            plt.savefig(name_fig)
        
        plt.show()
        plt.close()


### Examples ###
# 1) Print a kernel
#chkpt = '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/SF/training_VGG_16_encoder_decoder.ckpt-293318'
#vars_chkpt = tf.train.list_variables(chkpt)
#print('Variables found in the chkpt file: {}'.format(vars_chkpt))
#print('Kernel of the first convolutional layer:')
#kernel = tf.train.load_variable(chkpt, 'VGG_16_encoder_decoder/conv1_1/kernel')






###################################################################################
#def plot_timeseries_from_video(vid, features = ['SIZE', 'SIZE_ACR', 'NACR']):
#    timeseries, sample_time = get_timeseries_from_video(vid, features)
#    (n,p) = np.shape(timeseries)
#    fig, axes = plt.subplots(n, 1)
#    fig.subplots_adjust(hspace=0.3, wspace=0.3)
#    for i, ax in enumerate(axes.flat):
#        ax.plot(sample_time, timeseries[i,:])
#        ax.set_ylabel(features[i])
#        ax.set_xlabel('Time before event (in minutes)\n Flare class {}'.format(vid['flare_event']['event_class']))
#    plt.show()
#
#
#    
#
#
#def plot_pictures(pics, labels, preds = None, segs='Bp,Br,Bt,Dopplergram,Continuum,Magnetogram'):
#    assert len(pics) == len(labels) > 0 and ((preds is None) or len(preds) == len(labels))
#    segs = re.split(' ?, ?', segs)
#    pics_shape = np.shape(pics)
#    if(len(pics_shape) != 4):
#        print('Shape of pictures must be nb_pics x height x weight x nb_channels')
#        return False
#    nb_channels = pics_shape[3]
#    nb_pics = pics_shape[0]
#    assert len(segs) == nb_channels
#
#    for k in range(nb_pics):
#        # Print every channels of an image
#        pic = pics[k]
#        if(nb_channels == 1):
#            plt.imshow(pic[:,:,0])
#            if(preds is None):
#                plt.xlabel('Label {0}'.format(labels[k]))
#            else:
#                plt.xlabel('Label {0}, Pred {1}'.format(preds[k]))
#            plt.xticks([])
#            plt.yticks([])
#        else:
#            fig, axes = plt.subplots(int(nb_channels/3)+(nb_channels%3>0), \
#                                     int(3*(nb_channels>=3)+nb_channels*(nb_channels<3)),\
#                                     figsize=(15, 7.5))
#            for i, ax in enumerate(axes.flat):
#                if(i < nb_channels):
#                    # Plot image.
#                    ax.imshow(pic[:,:,i])
#                    if(preds is None):
#                        ax.set_xlabel('{0}, label {1}'.format(segs[i], labels[k]))
#                    else:
#                        ax.set_xlabel('{0}, \n Label {1}, pred {2}'.format(segs[i], labels[k], preds[k]))
#                    # Remove ticks from the plot.
#                    ax.set_xticks([])
#                    ax.set_yticks([])
#            fig.subplots_adjust(hspace=0.05, wspace=0.1)
#
#        plt.show()

