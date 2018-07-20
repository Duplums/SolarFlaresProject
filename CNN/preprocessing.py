'''
This class aims to manage the data stored on the disk and to preprocess it.
There is 2 different types of data:
    * The ones coming from the JSOC data base (HDF5 files)
    * The ones coming from MNIST, CIFAR, ImageNet, ... already preprocessed ! (HDF5 files )
      This class preprocesses only the data coming from the JSOC data base.
'''



import os, math, drms, traceback, re
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py as h5
import numpy as np

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
    # Directory where we save features (if needed)
    features_dir = None
    def __init__(self, db, paths_to_file = [], subsampling = 1, 
                 nb_classes = 2, pic_size = (256, 512), 
                 resize_method = 'LIN_RESIZING', segs = ['Bp', 'Br', 'Bt'],
                 time_step = 60, features_dir=None):
        
        self.paths_to_file = list(paths_to_file)
        self.database = db
        if(db in {'MNIST', 'CIFAR-10', 'IMG_NET'}):
            print('Warning: no preprocessing for this data base. We assert that the files are organized as follows:\n')
            print('\t/features[dataset]\n')
            print('\t/labels[dataset]\n')
        elif(db == 'SF'):
            if(resize_method in {'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}):
                self.subsampling = subsampling
                self.nb_classes = nb_classes
                self.pic_size = tuple(pic_size)
                self.resize_method = resize_method
                self.segs = segs
                self.time_step = time_step
                self.features_dir= features_dir
            else:
                print('Resizing method unknown.')
                raise
        else:    
            print('Data base unknown.')
            raise
            
    def set_files(self, paths_to_file):
        self.paths_to_file = list(paths_to_file)
    
    def add_files(self, paths_to_file):
        self.paths_to_file += list(paths_to_file)
    
    def clear_files(self):
        self.paths_to_file = []
    
    # Assigns a label (int number) associated to a flare class.
    # This label depends of the number of classes.
    def _label(self, flare_class):
        if(self.nb_classes == 2):
            return int(flare_class[0] >= 'M')
        else:
            print('Number of classes > 2 case : not yet implemented')
            raise
    
    # Returns the maximum size of pictures found in all files given to the preprocessor
    def get_max_size(self):
        if(self.database == 'SF'):
            max_size = [-math.inf, -math.inf]
            for file_path in self.paths_to_file:
                if(os.path.isfile(file_path)):
                    try:
                        with h5.File(file_path, 'r') as db:
                            for vid_key in db.keys():
                                for frame_key in db[vid_key]:
                                    if('channels' in db[vid_key][frame_key].keys() and
                                       len(db[vid_key][frame_key]['channels'].shape) >=2):
                                        max_size[0] = max(max_size[0], db[vid_key][frame_key]['channels'].shape[0])
                                        max_size[1] = max(max_size[1], db[vid_key][frame_key]['channels'].shape[1])
                                    
                            
                            
                    except:
                        print('Impossible to get the size of pictures in {}'.format(file_path))
                        print(traceback.format_exc())
                        
                else:
                    print('Warning: {} is not a file. Ignored'.format(file_path))
            return max_size
        else:
            print('Wrong data base (got {} instead of \'SF\')'.format(self.database))
            return None
    # check 'NaN' in a frame that can have one or multiple channels. Returns
    # a clean frame. INPUT: np array with shape (h, w, c)
    @staticmethod
    def _check_nan(frame):
        shape = frame.shape
        new_frame = None
        if(len(shape) != 3):
            print('Shape of frame must be (h, w, c) (got {})'.format(shape))
            raise        
        for c in range(shape[2]):
            pic = frame[:,:,c]
            if(np.any(np.isnan(pic))):
                print('Warning: NaN found in a frame. Trying to erase them...')
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
            if(new_frame is None):
                new_frame = np.zeros(pic.shape+(shape[2],), dtype=np.float32)
            try:
                new_frame[:,:,c] = pic
            except:
                print('All channels are no coherents')
                print('Previous channel:')
                plt.imshow(frame[:,:,c-1])
                plt.show()
                print('New channel:')
                plt.imshow(frame[:,:,c])
                plt.show()
                print(traceback.format_exc())
                raise
        return new_frame
    
    def _extract_frame(self, frame, frame_segs):
        nb_channels = len(self.segs)
        shape_frame = frame.shape[0:2] + (nb_channels,)
        frame_tensor = np.zeros(shape_frame, dtype=np.float32)
        channel_counter = 0
        # Considers only certain segments
        for k in range(len(frame_segs)):
            if(frame_segs[k].decode() in self.segs):
                frame_tensor[:,:,channel_counter] = frame[:,:,k]
                channel_counter += 1
        # Checks 'NaN' (be careful ,the size might change)
        frame_tensor = self._check_nan(frame_tensor)
        shape_frame = frame_tensor.shape
        # Resizes the frame and returns a Tensor
        if(self.resize_method == 'LIN_RESIZING'):
            return tf.image.resize_images(tf.image.per_image_standardization(frame_tensor), 
                                          size=self.pic_size, method=tf.image.ResizeMethod.BILINEAR)
        elif(self.resize_method == 'QUAD_RESIZING'):
            return tf.image.resize_images(tf.image.per_image_standardization(frame_tensor), 
                                          size=self.pic_size, method=tf.image.ResizeMethod.BICUBIC)
        elif(self.resize_method == 'ZERO_PADDING'):
            pad_x_up = math.floor((self.pic_size[0]-shape_frame[0])/2.0)
            pad_x_down = self.pic_size[0] - (pad_x_up + shape_frame[0])
            pad_y_left = math.floor((self.pic_size[1]-shape_frame[1])/2.0)
            pad_y_right = self.pic_size[1] - (pad_y_left + shape_frame[1])
            return tf.pad(tf.image.per_image_standardization(frame_tensor), 
                          [[pad_x_up, pad_x_down], [pad_y_left, pad_y_right], [0, 0]])
        else:
            print('Error: unknown resizing method')
            raise

    
    
    # Extract the data from the list of files according to the parameters set.
    # OUTPUT : a tuple (features, labels) of 2 Tensors of dims and types:
    #   * features.shape == (nb_samples, width, height, nb_channels)
    #   * labels.shape == (nb_samples, 1)
    #   * features.dtype == float32
    #   * labels.dtype == int32
    
    def extract_features(self, saving = False, retrieve = False):
        features = []
        labels = []
        feature_counter= 0
        for file_path in self.paths_to_file:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        if(self.database == 'SF'):
                            curr_features = None
                            curr_labels = None
                            if((retrieve or saving) and self.features_dir is not None):
                                features_name =  re.sub('.hdf5', '', os.path.basename(file_path)) + '_features.npy'
                                labels_name = re.sub('.hdf5', '', os.path.basename(file_path)) + '_labels.npy'
                                features_path = os.path.join(self.features_dir, features_name)
                                labels_path = os.path.join(self.features_dir, labels_name)
                            # First, check if we can retrieve features
                            if(retrieve and os.path.isfile(features_path) 
                                and os.path.isfile(labels_path)):
                                try:
                                    curr_features = np.load(features_path)
                                    curr_labels = np.load(labels_path)
                                    feature_counter += len(curr_features)
                                    print('Features retrieved from {}, {}'.format(features_name, labels_name))
                                except:
                                    print('Error while retrieving features.')
                                    print(traceback.format_exc())
                                    raise
                            if(curr_features is None or curr_labels is None):
                                # Takes each video in each file, down samples the nb of frames
                                # in each one and resize the frames according to 'resize_method'
                                curr_features = []
                                curr_labels = []
                                for vid_key in db.keys():
                                    frame_counter = 0
                                    label = self._label(db[vid_key].attrs['event_class'])
                                    for frame_key in db[vid_key].keys():
                                        # subsample the video
                                        if(frame_counter % self.subsampling == 0):
                                            if('channels' in db[vid_key][frame_key].keys()):
                                                # Tensor that represents a frame
                                                frame_tensor = self._extract_frame(db[vid_key][frame_key]['channels'], db[vid_key][frame_key].attrs['SEGS'])                              
                                                curr_features += [frame_tensor]
                                                curr_labels += [label]
                                print('Features extracted from file {}.'.format(file_path))
                                if(saving):
                                    try:
                                        np.save(features_path, np.array(curr_features, dtype=np.float32))
                                        np.save(labels_path, np.array(curr_labels, dtype=np.float32))
                                    except:
                                        print('Impossible to save the features extracted.')
                                        print(traceback.format_exc())
                            features += curr_features
                            labels += curr_labels
                                
                        else:
                            curr_features = np.array(db['features'], dtype=np.float32)
                            curr_labels = np.array(db['labels'], dtype=np.int32)
                            feature_counter += len(curr_features)
                            if(len(curr_features) == len(curr_labels)):
                                if(features is None):
                                    features = curr_features
                                    labels = curr_labels
                                else:
                                    features = np.concatenate((features, curr_features))
                                    labels = np.concatenate((labels, curr_labels))
                            else:
                                print('Features and labels have different lengths in file {}. Ignored'.format(file_path))
                except:
                    print('Impossible to extract features from {}'.format(file_path))
                    print(traceback.format_exc())
            else:
                print('File {} does not exist. Ignored'.format(file_path))
       
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
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
        
