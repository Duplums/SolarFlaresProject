
'''
This class aims to create a list of files where we can find the data and to manage
the computer memory. The data memory size (in MB) is given and should not be exceeded.
It also builds the input pipeline for the whole TensorFlow computation and thus 
a graph is needed.
'''

import os, re, pickle, traceback, math, drms
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py as h5
    
    
class Data_Gen:
    
    graph = None
    paths_to_file = None # array of paths where we can find the data
    max_pic_size = None
    size_of_files = None
    memory_size = None
    features_dir = None
    data_dims = None 
    segs = None
    database_name = None
    time_step = None
    dataset = None
    tf_device = None
    nb_classes = None
    subsampling = None
    resize_method = None
    data_iterator = None
    training_mode = None
    
    def __init__(self, data_name, config, graph, tf_device = '/cpu:0',
                 training=True, max_pic_size=None):
        assert data_name in {'SF', 'MNIST', 'CIFAR-10', 'IMG_NET'}
        
        self.paths_to_file = []
        self.size_of_files = []
        self.memory_size =  config['batch_memsize']
        self.data_dims = config['data_dims']
        self.database_name = data_name
        self.training_mode = training
        self.nb_classes = config['nb_classes']
        self.segs = config['segs']
        self.batch_size = config['batch_size']
        self.max_pic_size = max_pic_size
        self.tf_device = tf_device
        self.graph = graph
        
        if(data_name == 'SF'):
            assert config['resize_method'] in {'NONE', 'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}
            self.features_dir = config['features_dir']
            self.resize_method = config['resize_method']
            self.time_step = config['time_step']
        if(training):
            self.subsampling = config['subsampling']
            paths = config['training_paths']
        else:
            self.subsampling = 1
            paths = config['testing_paths']

        if(data_name in {'MNIST', 'CIFAR-10', 'IMG_NET'}):
            print('Warning: no preprocessing for this data base. We assert that the files are organized as follows:\n')
            print('\t/features[dataset]\n')
            print('\t/labels[dataset]\n')

        # First checks
        for path in paths:
            if os.path.exists(path):
                for file in os.listdir(path):   
                    path_to_file = os.path.join(path, file)
                    if(os.path.isfile(path_to_file)):
                        size = os.path.getsize(path_to_file)/(1024*1024)
                        if(size <= self.memory_size):
                            self.paths_to_file += [path_to_file]
                            self.size_of_files += [size/float(self.subsampling)]
                        else:
                            print('File {} will not fit in memory. Ignored'.format(path_to_file))
                    else:
                        print('Directory \'{}\' ignored.'.format(path_to_file))
            else:
                print('Path {} does not exist. Ignored'.format(path))
        
        if(self.max_pic_size is None):
            print('Computing the maximum of picture size...')
            self.max_pic_size = self.get_max_size()
        if(self.max_pic_size[0] < 0 or self.max_pic_size[1] < 0):
            print('Warning: maximum size unknown ({}), could lead to error'.format(self.max_pic_size))
        else:
            print('Maximum size found : {}'.format(self.max_pic_size))
    
     # Returns the maximum size of pictures found in all files
    def get_max_size(self):
        max_size = [-math.inf, -math.inf]
        for file_path in self.paths_to_file:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        if(self.database_name == 'SF'):
                            for vid_key in db.keys():
                                for frame_key in db[vid_key]:
                                    if('channels' in db[vid_key][frame_key].keys() and
                                       len(db[vid_key][frame_key]['channels'].shape) >=2):
                                        max_size[0] = max(max_size[0], db[vid_key][frame_key]['channels'].shape[0])
                                        max_size[1] = max(max_size[1], db[vid_key][frame_key]['channels'].shape[1])
                        else:
                            max_size[0] = max(max_size[0], db['features'].shape[1])
                            max_size[1] = max(max_size[1], db['features'].shape[2])
                except:
                    print('Impossible to get the size of pictures in {}'.format(file_path))
                    print(traceback.format_exc())
                    raise
            else:
                print('Warning: {} is not a file. Ignored'.format(file_path))
        return max_size
    
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
        if(self.database_name != 'SF'):
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
        
    # Assigns a label (int number) associated to a flare class.
    # This label depends of the number of classes.
    def _label(self, flare_class):
        if(self.nb_classes == 2):
            return int(flare_class[0] >= 'M')
        else:
            print('Number of classes > 2 case : not yet implemented')
            raise
                
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
        return frame_tensor
        
    # Extract the data from the list of files according to the parameters set.
    # OUTPUT : 2 lists that contains pictures of possibly various sizes and 
    # the labels associated.
    def _extract_data(self, files_to_extract, saving = False, retrieve = False):
        features = []
        labels = []
        for file_path in files_to_extract:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        print('Beginning to extract data from {}'.format(file_path))
                        if(self.database_name == 'SF'):
                            curr_features = None
                            curr_labels = None
                            if((retrieve or saving) and self.features_dir is not None):
                                features_name =  re.sub('.hdf5', '', os.path.basename(file_path)) + '_features.bin'
                                labels_name = re.sub('.hdf5', '', os.path.basename(file_path)) + '_labels.bin'
                                features_path = os.path.join(self.features_dir, features_name)
                                labels_path = os.path.join(self.features_dir, labels_name)
                            # First, check if we can retrieve features
                            if(retrieve and os.path.isfile(features_path) 
                                and os.path.isfile(labels_path)):
                                try:
                                    with open(features_path, 'rb') as f:
                                        curr_features = pickle.load(f)
                                    with open(labels_path, 'rb') as l:
                                        curr_labels = pickle.load(l)
                                    print('Data retrieved from {}, {}'.format(features_name, labels_name))
                                except:
                                    print('Error while retrieving features.')
                                    print(traceback.format_exc())
                                    raise
                            if(curr_features is None or curr_labels is None):
                                # Takes each video in each file and down samples the nb of frames
                                # and erase 'NaN'
                                curr_features = []
                                curr_labels = []
                                for vid_key in db.keys():
                                    frame_counter = 0
                                    label = self._label(db[vid_key].attrs['event_class'])
                                    for frame_key in db[vid_key].keys():
                                        # subsample the video
                                        if(frame_counter % self.subsampling == 0):
                                            if('channels' in db[vid_key][frame_key].keys()):
                                                frame_tensor = self._extract_frame(db[vid_key][frame_key]['channels'], db[vid_key][frame_key].attrs['SEGS'])                              
                                                curr_features += [frame_tensor]
                                                curr_labels += [label]
                                print('Data extracted from file {}.'.format(file_path))
                                if(saving):
                                    try:
                                        with open(features_path, 'wb') as f:
                                            pickle.dump(curr_features, f)
                                        with open(labels_path, 'wb') as l:
                                            pickle.dump(curr_labels, l)
                                    except:
                                        print('Impossible to save the data extracted.')
                                        print(traceback.format_exc())
                                        raise
                            features += curr_features
                            labels += curr_labels
                                
                        else:
                            curr_features = np.array(db['features'], dtype=np.float32)
                            curr_labels = np.array(db['labels'], dtype=np.int32)
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
        return (features, labels)
    
    
    def gen_batch_dataset(self, 
                   save_extracted_data = False, 
                   retrieve_data = False,
                   take_random_files = False):
        
        batch_mem = 0
        if(take_random_files):
            files_index = np.random.permutation(len(self.size_of_files))
        else:
            # Heuristic to know which files we will consider
            files_index = np.argsort(self.size_of_files)
        
        # Store the file paths according to their size (by taking random files or 
        # by using a heuristic inspired from the knapspack pb).
        files_in_batch= []
        counter = 0
        for k in files_index:
            if self.size_of_files[k] + batch_mem <= self.memory_size:
                files_in_batch += [self.paths_to_file[k]]
                batch_mem += self.size_of_files[k]
                counter += 1
            else:
                break
        self.size_of_files = [s for k, s in enumerate(self.size_of_files) if k not in files_index[:counter]]
        self.paths_to_file = [s for k, s in enumerate(self.paths_to_file) if k not in files_index[:counter]]
        
        print('Files to be loaded in memory : ')
        for f in files_in_batch:
            print('\t - {} => {}MB'.format(f, math.ceil(os.path.getsize(f)/(1024.0*1024))))
        # Load the data in memory
        (features, labels) = self._extract_data(files_in_batch, save_extracted_data, retrieve_data)
        print('Data set loaded.')
        return (features, labels)
    
    #############################################################
    #                   TENSORFLOW GRAPH                        #
    #############################################################
    
     # Used as input for the TensorFlow pipeline
    @staticmethod
    def generator(features, labels):
        for k in range(len(features)):
            yield (features[k], labels[k])
    
    # We assume that 'tensor' is a picture with shapes
    # (h, w, c) and the key associated should be unique 
    # according to the size.
    def _get_key_from_tensor(self, tensor):
        return tf.cast(tf.shape(tensor)[0] +(max(self.max_pic_size)+1)*tf.shape(tensor)[1], tf.int64)
    
    def _shuffle_per_batch(self, dataset):
        return dataset.batch(self.batch_size).shuffle(self.batch_size)
   
    def _zero_padding(self, pic):
        pad_x_up = math.floor((self.max_pic_size[0]-pic.shape[0])/2.0)
        pad_x_down = self.max_pic_size[0] - (pad_x_up + pic.shape[0])
        pad_y_left = math.floor((self.max_pic_size[1]-pic.shape[1])/2.0)
        pad_y_right = self.max_pic_size[1] - (pad_y_left + pic.shape[1])
        
        return tf.pad(pic, [[pad_x_up, pad_x_down], [pad_y_left, pad_y_right], [0, 0]])
    
    
    def data_preprocessing(self):
        if(self.resize_method == 'NONE'):
            self.dataset = self.dataset.map(lambda pic, label : (tf.image.per_image_standardization(pic), label))
        
        elif(self.resize_method == 'LIN_RESIZING'):
            self.dataset = self.dataset.map(lambda pic, label : (tf.image.per_image_standardization(
                                                                    tf.image.resize_images(pic, 
                                                                                           size=self.data_dims[0:2], 
                                                                                           method=tf.image.ResizeMethod.BILINEAR)),
                                                                 label))
            
            
        elif(self.resize_method == 'QUAD_RESIZING'):
            self.dataset = self.dataset.map(lambda pic, label : (tf.image.per_image_standardization(
                                                                    tf.image.resize_images(pic, 
                                                                                           size=self.data_dims[0:2], 
                                                                                           method=tf.image.ResizeMethod.BICUBIC)),
                                                                 label))
        
        elif(self.resize_method == 'ZERO_PADDING'):
            self.dataset = self.dataset.map(lambda pic, label : (self._zero_padding(tf.image.per_image_standardization(pic)),
                                                                 label))
            
        else:
            print('Error: unknown resizing method')
            raise
    
    
    def create_tf_dataset_and_preprocessing(self, features, labels):
        print('Constructing the new TensorFlow input pipeline on device {}...'.format(self.tf_device))
        with self.graph.as_default(), tf.device(self.tf_device):
            self.dataset = tf.data.Dataset.from_generator(lambda: self.generator(features, labels),
                                                      output_types = (tf.float32, tf.int32),
                                                      output_shapes = (tf.TensorShape([None, None, self.data_dims[2]]),
                                                                       tf.TensorShape([None])))
            print('Data preprocessing...')
            self.data_preprocessing()
            print('Grouping pictures by size...')
            self.dataset = self.dataset.apply(tf.contrib.data.group_by_window(lambda pic, label : self._get_key_from_tensor(pic),
                                                                              lambda tensors : self._shuffle_per_batch(tensors),
                                                                              window_size=self.batch_size))
            self.data_iterator = self.dataset.make_one_shot_iterator()
            print('New TF Dataset created.')
    
    def get_next_batch(self):
        try:
            return self.data_iterator.get_next()
        except tf.errors.OutOfRangeError:
            return (None, None)
    
    
    

                
