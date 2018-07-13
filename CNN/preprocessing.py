'''
This class aims to manage the data stored on the disk and to preprocess it.
There is 2 different types of data:
    * The ones coming from the JSOC data base (HDF5 files)
    * The ones coming from MNIST, CIFAR, ImageNet, ... already preprocessed ! (HDF5 files )
      This class preprocesses only the data coming from the JSOC data base.
'''



import os, math, drms, traceback
import h5py as h5
import numpy as np
import skimage.transform as sk

class Preprocessor:
    
    # Array of files we will work with
    paths_to_file = None
    # data base considered (in {'SF', 'MNIST', 'CIFAR', 'IMG_NET'})
    database = None
    # Subsampling the videos for the 'SF' data base
    subsampling = None
    # How many labels do we consider for the 'SF' data base ?
    nb_classes = None
    # What is the common size that we want to rescale pictures for the 'SF' data base ?
    pic_size = None
    # How do we manage the size of pictures in each video for the 'SF' data base ?
    # Could be {'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}
    resize_method = None
    # What segments do we consider for the 'SF' data base
    segs = None
    # What is the time step used for each video (in minutes) ?
    time_step = None
    
    def __init__(self, db, paths_to_files = [], subsampling = 1, 
                 nb_classes = 2, pic_size = (256, 512), 
                 resize_method = 'LIN_RESIZING', segs = ['Bp', 'Br', 'Bt'],
                 time_step = 60):
        
        self.paths_to_files = list(paths_to_files)
        self.database = db
        if(db in {'MNIST', 'CIFAR', 'IMG_NET'}):
            print('Warning: no preprocessing for this data base. We assert that the files are organized as follows:\n')
            print('\t/features/[dataset]\n')
            print('\t/labels/[dataset]\n')
        elif(db == 'SF'):
            if(resize_method in {'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}):
                self.subsampling = subsampling
                self.nb_classes = nb_classes
                self.pic_size = tuple(pic_size)
                self.resize_method = resize_method
                self.segs = segs
                self.time_step = time_step
            else:
                print('Resizing method unknown.')
                raise
        else:    
            print('Data base unknown.')
            raise
            
    def set_files(self, paths_to_files):
        self.path_to_files = list(paths_to_files)
    
    def add_files(self, paths_to_files):
        self.paths_to_file += list(paths_to_files)
    
    def clear_files(self):
        self.path_to_files = []
   
    # Normalizes a picture
    @staticmethod
    def _normalize(pic):
        m = np.mean(pic)
        sd = np.std(pic)
        return (pic - m)/sd
    
    # Assigns a label (int number) associated to a flare class.
    # This label depends of the number of classes.
    def _label(self, flare_class):
        if(self.nb_classes == 2):
            return int(flare_class[0] >= 'M')
        else:
            print('Number of classes > 2 case : not yet implemented')
            raise
    
    # Resizes a picture using linear or quadratic interpolation or zero-padding.
    def _resize(self, pic):
        if(np.any(np.isnan(pic))):
            print('Warning: NaN found in a picture. Trying to erase them...')
            where_is_nan = np.argwhere(np.isnan(pic))
            nan_up_left_corner = (min(where_is_nan[:,0]), min(where_is_nan[:,1]))
            nan_down_right_corner = (max(where_is_nan[:,0]), max(where_is_nan[:,1]))
            # Select only the biggest rectangle that does not contain 'NaN'
            pic_shape = pic.shape
            right_split_pic_width = nan_up_left_corner[1]
            left_split_pic_width = pic_shape[1] - nan_down_right_corner[1]
            if(right_split_pic_width > left_split_pic_width):
                # conserve only the right picture's part
                pic = pic[:, 0:nan_up_left_corner[1]]
            else:
                # otherwise, conserve the other part
                pic = pic[:, nan_down_right_corner[1]+1:]
            pic_reshape = pic.shape
            
            if(np.any(np.isnan(pic))):
                print('Impossible to erase NaN.')
            else:
                print('NaN erased. Reshape operation: {} --> {}'.format(pic_shape, pic_reshape))
            
        if(self.resize_method == 'LIN_RESIZING'):
            sk.resize(pic, self.pic_size, order=1, preserve_range=True)
        elif(self.resize_method == 'QUAD_RESIZING'):
            return sk.resize(pic, self.pic_size, order=2, preserve_range=True)
        else:
            current_pic_shape = pic.shape
            resized_pic = np.zeros(self.pic_size, dtype=np.float32)
            h_marge = math.floor((self.pic_size[0]-current_pic_shape[0])/2.0)
            w_marge = math.floor((self.pic_size[1]-current_pic_shape[1])/2.0)
            
            if(h_marge < 0 or w_marge < 0):
                print('Warning: zero-padding is used for picture of shape {} (common size is {})\n'.format(current_pic_shape, self.pic_size))
                print('Picture is resized instead (using linear interpolation)')
                return sk.resize(pic, self.pic_size, order=1, preserve_range=True)
            
            resized_pic[h_marge:h_marge+current_pic_shape[0],w_marge:w_marge+current_pic_shape[1]] = pic
            return resized_pic
    
    
    # Extract the data from the list of files according to the parameters set.
    # OUTPUT : a tuple (features, labels) of 2 numpy arrays of dims and types:
    #   * features.shape == (nb_samples, width, height, nb_channels)
    #   * labels.shape == (nb_samples, 1)
    #   * features.dtype == float32
    #   * labels.dtype == int32
    
    def extract_features(self):
        features = None
        labels = None
        if(self.database == 'SF'):
            nb_channels = len(self.segs)
            features = np.zeros((0,)+self.pic_size+(nb_channels,), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int32) 
        
        for file_path in self.paths_to_file:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        if(self.database == 'SF'):
                            # Takes each video in each file, down samples the nb of frames
                            # in each one and resize the frames according to 'resize_method'
                            for vid_key in db.keys():
                                frame_counter = 0
                                for frame_key in db[vid_key].keys():
                                    # subsample the video
                                    if(frame_counter % self.subsampling == 0):
                                        frame_tensor = np.zeros((1,)+self.pic_size+(nb_channels,), dtype=np.float32)
                                        frame_segs = db[vid_key][frame_key].attrs['SEGS']
                                        vid_flare_class = db[vid_key].attrs['event_class']
                                        label = self._label(vid_flare_class)
                                        channel_counter = 0
                                        for k in range(len(frame_segs)):
                                            # consider only certain segments
                                            if(frame_segs[k] in self.segs):
                                                frame_tensor[:,:,:,channel_counter] = self._normalize(self._resize(db[vid_key][frame_key]['channels'][:,:,k]))
                                                channel_counter += 1
                                        ## IT HAS BE IMPROVED ##
                                        features = np.concatenate((features, frame_tensor), dtype=np.float32)
                                        labels = np.concatenate((labels, label), dtype=np.int32)
                        else:
                            curr_features = np.array(db['features'], dtype=np.float32)
                            curr_labels = np.array(db['labels'], dtype=np.int32)
                            if(len(curr_features) == len(curr_labels)):
                                if(features is None):
                                    features = curr_features
                                    labels = curr_labels
                                else:
                                    features = np.concatenate((features, curr_features), dtype=np.float32)
                                    labels = np.concatenate((labels, curr_labels), dtype=np.int32)
                            else:
                                print('Features and labels have different lengths in file {}. Ignored'.format(file_path))  
                except:
                    print('Impossible to extract features from {}'.format(file_path))
                    print(traceback.format_exc())
            else:
                print('File {} does not exist. Ignored'.format(file_path))
        
        return (features, labels)
    
    
    # Takes a video and a list of scalars as input and returns the corresponding
    # time series.
    def _extract_timeseries_from_video(self, vid, scalars):
        res = np.zeros((len(scalars), len(vid)), dtype=np.float32)
        sample_time = np.zeros(len(vid), dtype=np.float32)
        tf = drms.to_datetime(vid.attrs['end_time'])
        i = 0
        j = 0
        for frame_key in vid.keys():
            ti = drms.to_datetime(vid[frame_key].attrs['T_REC'])
            sample_time[j] = (tf - ti).total_seconds()/60
            for scalar in scalars:
                res[i,j] = vid[frame_key].attrs[scalar]
                i += 1
            j += 1
            i = 0
        return res, sample_time
    
    # Extracts some scalars from video that evolves according to the time (ex: SIZE of a frame).
    # The scalar must be in the frame attributes of a video. These time series are concatenate
    # in one list for every videos. 'tstart' and 'tend' are used to know when the time series
    # begin and when they end (from an event, time reversed). If values are missing (<5% by default), 
    # they are interpolated.
    def extract_timeseries(self, scalars, tstart, tend, loss=0.05):
        if(self.database != 'SF'):
            print('The data base must be from JSOC.')
            return None
        
        nb_frames = int((tend - tstart)/self.time_step) + 1
        nb_scalars = len(scalars)
        sample_time = np.linspace(tstart, tend, nb_frames)
        res = []
        print('{} frames are considered from {}min before a solar eruption to {}min.'.format(nb_frames, tstart, tend))
        print('INFO: a linear interpolation is used to reconstruct the time series.')
        for file_path in self.paths_to_file:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        for vid_key in db.keys():
                            vid_time_series, vid_sample_time = self._extract_timeseries_from_video(db[vid_key], scalars)
                            i_start = np.argmin(abs(sample_time - tstart))
                            i_end = np.argmin(abs(sample_time - tend))
                            if(np.any(np.isnan(vid_time_series))):
                                print('Video {} ignored because the time series associated contains \'NaN\'.'.format(vid_key))
                            elif(abs(sample_time[i_start] - tstart) <= self.time_step):
                                nb_frames_in_vid = i_end - i_start + 1
                                if(1 - nb_frames_in_vid/nb_frames <= loss):
                                    res_vid = np.zeros((nb_scalars, nb_frames), dtype=np.float32)
                                    for k in range(nb_scalars):
                                        res_vid[k,:] = np.interp(sample_time, vid_sample_time, vid_time_series[k,:])
                                    res += [res_vid]
                            
                except:
                    print('Impossible to extract time series from file {}'.format(file_path))
                    print(traceback.format_exc())
            else:
                print('File {} does not exist. Ignored'.format(file_path))

            return np.array(res, dtype=np.float32), sample_time



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
        
