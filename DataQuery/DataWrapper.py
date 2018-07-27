import cv2
import numpy as np
import traceback
import h5py as h5


class SF_video:






#def picture_show(pics, segs = 'Bp,Bp_err,Br,Br_err,Bt,Bt_err,Dopplergram,magnetogram,continuum'):
#    segs = re.split(' ?, ?', segs)
#    flare_class = pics['flare_events'][0]['event_class']
#    peak = pics['flare_events'][0]['peak_time']
#    trec = pics['header']['T_REC']
#    noaa = pics['header']['NOAA_ARS']
#    harp = pics['header']['HARPNUM']
#    nb_channels = len(segs)
#    # Print every channels of an image
#    fig, axes = plt.subplots(int(nb_channels/3)+(nb_channels%3>0), \
#                             int(3*(nb_channels>=3)+nb_channels*(nb_channels<3)),\
#                             figsize=(15, 7.5))
#    for i, ax in enumerate(axes.flat):
#        if(i < nb_channels):
#        # Plot image.
#            ax.imshow(pics[segs[i]])
#            ax.set_xlabel('{}'.format(segs[i]))
#            # Remove ticks from the plot.
#            ax.set_xticks([])
#            ax.set_yticks([])
#    fig.suptitle('AR {} (HARP {}) taken on {}\n with peak flux magnitude {}\n(peak time: {})'.\
#                 format(noaa, harp, trec, flare_class, peak), fontsize=14, fontweight='bold')
#    fig.subplots_adjust(hspace=0.1, wspace=0.1)
#    plt.show()











    
