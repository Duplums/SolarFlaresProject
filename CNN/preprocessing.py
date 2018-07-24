'''
This class aims to manage the data stored on the disk and to preprocess it.
There is 2 different types of data:
    * The ones coming from the JSOC data base (HDF5 files)
    * The ones coming from MNIST, CIFAR, ImageNet, ... already preprocessed ! (HDF5 files )
      This class preprocesses only the data coming from the JSOC data base.
'''


#############################
#       TEMP FUNCS          #
#############################






#
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
        
