
'''
This class aims to create a list of files where we can find the data and to manage
the computer memory. The data memory size (in MB) is given and should not be exceeded.
It also builds the input pipeline for the whole TensorFlow computation and thus 
a graph is needed.
'''

import os, re, pickle, traceback, math, drms, sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import skimage.transform as sk
import tensorflow as tf
import numpy as np
import h5py as h5
    
    
class Data_Gen:
    
    graph = None
    main_path = None
    paths_to_file = None # array of paths where we can find the data
    output_features_dir = None 
    output_features = None
    output_labels = None    
    max_pic_size = None
    size_of_files = None
    memory_size = None
    num_threads = None
    prefetch_buffer_size = None
    input_features_dir = None
    data_dims = None 
    segs = None
    database_name = None
    time_step = None
    tf_device = None
    nb_classes = None
    subsampling = None
    resize_method = None
    dataset = None
    data_iterator = None
    seq_length = None
    seq_length_iterator = None
    training_mode = None
    
    def __init__(self, data_name, config, graph = None, tf_device = '/cpu:0',
                 training=True, max_pic_size=None):
        assert data_name in {'SF', 'SF_LSTM', 'MNIST', 'CIFAR-10', 'IMG_NET'}
 
        self.paths_to_file = []
        self.size_of_files = []
        self.memory_size =  config['batch_memsize']
        self.data_dims = config['data_dims']
        self.database_name = data_name
        self.training_mode = training
        self.nb_classes = config['nb_classes']
        self.batch_size = config['batch_size']
        self.num_threads = config['num_threads']
        self.prefetch_buffer_size = config['prefetch_buffer_size']
        self.max_pic_size = max_pic_size
        self.tf_device = tf_device
        self.graph = graph
        self.subsampling = 1
        
        if(data_name == 'SF'):
            assert config['resize_method'] in {'NONE', 'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}
            assert os.path.isdir(config['input_features_dir'])
            assert os.path.isdir(config['output_features_dir'])
            self.segs = config['segs']
            self.input_features_dir = config['input_features_dir']
            self.output_features_dir = config['output_features_dir']
            self.resize_method = config['resize_method']
            self.time_step = config['time_step']
            self.output_features = {}
            self.output_labels = {}
            self._output_features_part_counter = 0
            if(training):
                self.subsampling = config['subsampling']

        if(training):
            self.main_path = config['training_paths']
        else:   
            self.main_path = config['testing_paths']

        if(data_name in {'MNIST', 'CIFAR-10', 'IMG_NET'}):
            print('Warning: no preprocessing for this data base. We assert that the HDF5 files are organized as follows:\n')
            print('\t/features[dataset]\n')
            print('\t/labels[dataset]\n')
        
        if(data_name == 'SF_LSTM'):
            print('Warning: we assume that each HDF5 file is organized as follows:\n')
            print('\t/features\n')
            print('\t\t meta(i) : numpy array [N_FRAMES(i), N_FEATURES(i)] (i in [1,N])\n')
            print('\t/labels\n')
            print('\t\t meta(i) : labels(i) (i in [1,N])\n')
        
        # First checks
        self.init_paths_to_file()
        
        if(self.max_pic_size is None and data_name == 'SF'):
            print('Computing the maximum of picture size...')
            self.max_pic_size = self.get_max_size()
        if(self.max_pic_size is None or max_pic_size[0] < 0 or self.max_pic_size[1] < 0):
            print('Warning: maximum size unknown ({}), could lead to error'.format(self.max_pic_size))
        else:
            print('Maximum size found : {}'.format(self.max_pic_size))
    
    def init_paths_to_file(self):
         for path in self.main_path:
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
    @staticmethod
    def _extract_timeseries_from_video(vid, scalars, channels):
        res = [[] for k in range(len(scalars))]
        sample_time = []
        tf = drms.to_datetime(vid.attrs['end_time'])
        first_frame = None
        for frame_key in sorted(list(vid.keys()), key=lambda frame_key : float(frame_key[5:])):
            if('channels' in vid[frame_key].keys() and
               len(vid[frame_key]['channels'].shape) == 3):
                ti = drms.to_datetime(vid[frame_key].attrs['T_REC'])
                sample_time += [(tf - ti).total_seconds()/60]
                if(first_frame is None):
                    first_frame = Data_Gen._extract_frame(vid[frame_key]['channels'], vid[frame_key].attrs['SEGS'], channels)
                i = 0
                for scalar in scalars:
                    if(scalar == 'RMS'):
                        l1_err = 0
                        try:
                            for c in range(first_frame.shape[2]):
                                if(vid[frame_key].attrs['SEGS'][c].decode() in channels):
                                        this_frame = Data_Gen._extract_frame(vid[frame_key]['channels'], vid[frame_key].attrs['SEGS'], channels)
                                        l1_err += np.sum(np.abs(sk.resize(this_frame[:,:,c], first_frame.shape[:2], preserve_range=True)-first_frame[:,:,c]))
                            res[i] += [l1_err/np.product(first_frame.shape[0:2])*len(channels)]
                        except:
                            sample_time = sample_time[:-1]
                            print('Frame {} not extracted.'.format(frame_key))
                            print(traceback.format_exc())
                    else:
                        res[i] += [vid[frame_key].attrs[scalar]]
                    i += 1
        return np.array(res), np.array(sample_time)
    
    # Extracts some scalars from video that evolve according to the time (ex: SIZE of a frame).
    # The scalar must be in the frame attributes of a video. These time series are concatenated
    # in one list for every videos. 'tstart' and 'tend' are used to know when the time series
    # begin and end (from an event, time reversed). If values are missing (<5% by default), 
    # they are interpolated.
    @staticmethod
    def extract_timeseries(paths_to_file = [], scalars = [], 
                           channels = ['Br', 'Bt', 'Bp'], 
                           time_step=60, tstart=0, tend=60*24, loss=0.05):
        nb_frames = int((tend - tstart)/time_step) + 1
        nb_scalars = len(scalars)
        sample_time = np.linspace(tstart, tend, nb_frames)
        res = []
        print('{} frames are considered from {}min before a solar eruption to {}min.'.format(nb_frames, tstart, tend))
        if(type(paths_to_file) is not list and os.path.isdir(paths_to_file)):
            print('Path to a directory. All the files inside are scanned')
            path = paths_to_file
            paths_to_file = []
            for file in os.listdir(path):
                if(os.path.isfile(os.path.join(path, file))):
                    paths_to_file += [os.path.join(path, file)]
        print('INFO: a linear interpolation is used to reconstruct the time series.')
        for file_path in paths_to_file:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        print('Analyzing file {}'.format(file_path))
                        for vid_key in db.keys():
                            vid_time_series, vid_sample_time = Data_Gen._extract_timeseries_from_video(db[vid_key], scalars, channels)
                            if(len(vid_sample_time) > 0):
                                i_start = np.argmin(abs(vid_sample_time - tstart))
                                i_end = np.argmin(abs(vid_sample_time - tend))
                                if(np.any(np.isnan(vid_time_series))):
                                    print('Video {} ignored because the time series associated contains \'NaN\'.'.format(vid_key))
                                elif(abs(sample_time[i_start] - tstart) <= time_step):
                                    nb_frames_in_vid = i_end - i_start + 1
                                    if(1 - nb_frames_in_vid/nb_frames <= loss):
                                        res_vid = np.zeros((nb_scalars, nb_frames), dtype=np.float32)
                                        for k in range(nb_scalars):
                                            res_vid[k,:] = np.interp(sample_time, vid_sample_time, vid_time_series[k,:])
                                        res += [res_vid]
                            
                except:
                    print('Impossible to extract time series from file {}'.format(file_path))
                    print(traceback.format_exc())
                    sys.exit(0)
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
    # Extract the frame numero from the frame key in each video of a SF.
    # We assume that each frame_key has the following format: 'frame{i}'
    # where {i} = the frame numero 
    @staticmethod
    def _to_frame_num(frame_key):
        nb = re.search(r'[1-9]?[0-9]?[0-9]', frame_key)
        if(nb is None):
            print('Impossible to parse the frame key {}'.format(frame_key))
            raise
        return int(nb.group())
    
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
                #plt.imshow(frame[:,:,c-1])
                #plt.show()
                print('New channel:')
                #plt.imshow(frame[:,:,c])
                #plt.show()
                print(traceback.format_exc())
                raise
        return new_frame
    
    @staticmethod
    def _extract_frame(frame, frame_segs, frame_final_segs):
        nb_channels = len(frame_final_segs)
        shape_frame = frame.shape[0:2] + (nb_channels,)
        frame_tensor = np.zeros(shape_frame, dtype=np.float32)
        channel_counter = 0
        # Considers only certain segments
        for k in range(len(frame_segs)):
            if(frame_segs[k].decode() in frame_final_segs):
                frame_tensor[:,:,channel_counter] = frame[:,:,k]
                channel_counter += 1
        # Checks 'NaN' (be careful ,the size might change)
        frame_tensor = Data_Gen._check_nan(frame_tensor)
        return frame_tensor
        
    # Extract the data from the list of files according to the parameters set.
    # OUTPUT : 2 lists that contains pictures of possibly various sizes and 
    # the labels associated. NOTE: if vid_infos is true, another list containing
    # video information relatively to the picture is added. Each meta data is 
    # formatted as : '{name_file}|{name_vid}|{active region size}|{frame_nb}'
    # !! CAREFUL: we assume that 'SIZE_ACR' is an attribute associated to EVERY frame
    # in the data base !! 
    def _extract_data(self, files_to_extract, saving = False, retrieve = False, 
                      vid_infos = False):
        features = []
        labels = []
        metadata = []
        for file_path in files_to_extract:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        print('Beginning to extract data from {}'.format(file_path))
                        if(self.database_name == 'SF'):
                            curr_features = None
                            curr_labels = None
                            curr_meta = None
                            if((retrieve or saving) and self.input_features_dir is not None):
                                features_name =  re.sub('.hdf5', '', os.path.basename(file_path)) + '_features.bin'
                                labels_name = re.sub('.hdf5', '', os.path.basename(file_path)) + '_labels.bin'
                                features_path = os.path.join(self.input_features_dir, features_name)
                                labels_path = os.path.join(self.input_features_dir, labels_name)
                                if(vid_infos):
                                    metadata_name = re.sub('.hdf5', '', os.path.basename(file_path)) + '_meta.bin'
                                    metadata_path = os.path.join(self.input_features_dir, metadata_name)
                            # First, check if we can retrieve features
                            if(retrieve and os.path.isfile(features_path) 
                                and os.path.isfile(labels_path)):
                                try:
                                    with open(features_path, 'rb') as f:
                                        curr_features = pickle.load(f)
                                    with open(labels_path, 'rb') as l:
                                        curr_labels = pickle.load(l)
                                    if(vid_infos):
                                      with open(metadata_path, 'rb') as m:
                                        curr_meta = pickle.load(m)
                                    print('Data retrieved from {}, {}'.format(features_name, labels_name))
                                    if(vid_infos):
                                        print('Meta data retrieved from {}'.format(metadata_name))
                                except:
                                    print('Error while retrieving features.')
                                    print(traceback.format_exc())
                                    raise
                            if(curr_features is None or curr_labels is None):
                                # Takes each video in each file and down samples the nb of frames
                                # and erase 'NaN'
                                curr_features = []
                                curr_labels = []
                                curr_meta= []
                                for vid_key in db.keys():
                                    frame_counter = 0
                                    label = self._label(db[vid_key].attrs['event_class'])
                                    meta = os.path.basename(file_path) + '|'+vid_key+'|'
                                    for frame_key in db[vid_key].keys():
                                        # subsample the video
                                        if(frame_counter % self.subsampling == 0):
                                            if('channels' in db[vid_key][frame_key].keys()):
                                                frame_tensor = Data_Gen._extract_frame(db[vid_key][frame_key]['channels'], db[vid_key][frame_key].attrs['SEGS'], self.segs)                              
                                                curr_features += [frame_tensor]
                                                curr_labels += [label]
                                                curr_meta += [meta+str(db[vid_key][frame_key].attrs['SIZE_ACR'])+'|'+str(self._to_frame_num(frame_key))]
                                print('Data extracted from file {}.'.format(file_path))
                                if(saving):
                                    try:
                                        with open(features_path, 'wb') as f:
                                            pickle.dump(curr_features, f)
                                        with open(labels_path, 'wb') as l:
                                            pickle.dump(curr_labels, l)
                                        print('Features saved in {}'.format(self.input_features_dir))
                                    except:
                                        print('Impossible to save the data extracted.')
                                        print(traceback.format_exc())
                                        raise
                            features += curr_features
                            labels += curr_labels
                            if(vid_infos):
                                metadata += curr_meta
                                
                        else:
                            if(self.database_name in {'MNIST', 'CIFAR-10', 'IMG_NET'}):
                                if(vid_infos):
                                    print('Warning: no meta data available for data base {}'.format(self.database_name))
                                curr_features = np.array(db['features'])
                                curr_labels = np.array(db['labels'])
                            elif(self.database_name == 'SF_LSTM'):
                                curr_features = []
                                curr_labels = []
                                for f_key in db['features'].keys():
                                    curr_features += [np.array(db['features'][f_key])]
                                    curr_labels += [np.array(db['labels'][f_key])]
                                if(vid_infos):
                                    metadata += list(db['features'].keys())
                            
                            if(len(curr_features) == len(curr_labels)):
                                features += curr_features
                                labels +=  curr_labels
                            else:
                                print('Features and labels have different lengths in file {}. Ignored'.format(file_path))
                except:
                    print('Impossible to extract features from {}'.format(file_path))
                    print(traceback.format_exc())
            else:
                print('File {} does not exist. Ignored'.format(file_path))
        
        if(vid_infos):
            return (features, labels, metadata)
        else:
            return (features, labels)
    
    
    def gen_batch_dataset(self, 
                   save_extracted_data = False, 
                   retrieve_data = False,
                   take_random_files = False,
                   get_metadata = False,
                   rm_paths_to_file = True):
        
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
        if(rm_paths_to_file):
            self.size_of_files = [s for k, s in enumerate(self.size_of_files) if k not in files_index[:counter]]
            self.paths_to_file = [s for k, s in enumerate(self.paths_to_file) if k not in files_index[:counter]]
        
        print('Files to be loaded in memory : ')
        for f in files_in_batch:
            print('\t - {} => {}MB'.format(f, math.ceil(os.path.getsize(f)/(1024.0*1024))))
        # Load the data in memory
        if(get_metadata):
            (features, labels, meta) = self._extract_data(files_in_batch, save_extracted_data, retrieve_data, True)
            print('Data set loaded.')
            return (features, labels, meta)
        
        (features, labels) = self._extract_data(files_in_batch, save_extracted_data, retrieve_data)
        print('Data set loaded.')
        return (features, labels)
    
    
    # We assume that each metadata are formatted as : '{name_file}|{name_vid}|{active region size}|{frame_nb}'
    # The active region size is added manually at the beginning of every features.
    def add_output_features(self, features, labels, metadata):
        assert len(features) == len(labels) == len(metadata)
        
        for k in range(len(metadata)):
            meta = metadata[k].decode()
            if(re.match('.+\|.+\|.+\|.+', meta)):
                meta_list = re.split(r'\|', meta)
                if(len(meta_list) == 4):
                    feature_key = meta_list[0]+'_'+meta_list[1]
                    frame_nb = int(meta_list[3])
                    size_acr = float(meta_list[2])
                    if(feature_key in self.output_features):
                        if(frame_nb not in self.output_features[feature_key]):
                            self.output_features[feature_key].update({frame_nb: [size_acr] + features[k]})
                        else:
                            print('Warning: frame {} already exists. Ignored'.format(frame_nb))
                    else:
                        self.output_features[feature_key] = {frame_nb: features[k]}
                        self.output_labels[feature_key] = labels[k]
                else:
                    print('Impossible to parse the metadata {}.Feature ignored'.format(meta))
            else:
                print('Unknown format for metadata {}. Feature ignored.'.format(meta))
    
    def dump_output_features(self):
        features_path = os.path.join(self.output_features_dir, 'output_features_part_{}.hdf5'.format(self._output_features_part_counter))
        with h5.File(features_path, 'w') as f:
            f.create_group('features')
            f.create_group('labels')
            for f_key in self.output_features:
                # Sort the dictionnary according to frame name
                self.output_features[f_key] = OrderedDict(sorted(self.output_features[f_key].items(), key=lambda t: t[0]))
                # Create a np array for the video (size nb_frames x n_features) and store it
                f['features'].create_dataset(f_key, data=np.array(list(self.output_features[f_key].values()), dtype=np.float32))
                f['labels'].create_dataset(f_key, data=self.output_labels[f_key], dtype=np.int32)
        
        # Clears the dictionnaries and increments the part counter
        self.output_features.clear()
        self.output_labels.clear()
        self._output_features_part_counter += 1

        
    #############################################################
    #                   TENSORFLOW GRAPH                        #
    #############################################################
    
     # Used as input for the TensorFlow pipeline
    @staticmethod
    def generator(features, labels, meta=None):
        if(meta is None):
            for k in range(len(features)):
                yield (features[k], labels[k])
        elif(meta is not None and len(meta) == len(features)):
            for k in range(len(features)):
                yield (features[k], labels[k], meta[k])
        else:
            print('Metadata have length {} but features have length {}.'.format(len(meta), len(features)))
            raise
    
    # We assume that 'tensor' is a picture with shapes
    # (h, w, c) and the key associated should be unique 
    # according to the size.
    def _get_key_from_tensor(self, tensor):
        return tf.cast(tf.shape(tensor)[0] +(max(self.max_pic_size)+1)*tf.shape(tensor)[1], tf.int64)
   
    def _zero_padding(self, pic):
        pad_x_up = math.floor((self.max_pic_size[0]-pic.shape[0])/2.0)
        pad_x_down = self.max_pic_size[0] - (pad_x_up + pic.shape[0])
        pad_y_left = math.floor((self.max_pic_size[1]-pic.shape[1])/2.0)
        pad_y_right = self.max_pic_size[1] - (pad_y_left + pic.shape[1])
        
        return tf.pad(pic, [[pad_x_up, pad_x_down], [pad_y_left, pad_y_right], [0, 0]])
    
    
    def data_preprocessing(self):
        if(self.resize_method == 'NONE'):
            self.dataset = self.dataset.map(lambda pic, label, *kw: (tf.image.per_image_standardization(pic), label, *kw), self.num_threads)
        
        elif(self.resize_method == 'LIN_RESIZING'):
            self.dataset = self.dataset.map(lambda pic, label, *kw: (tf.image.per_image_standardization(tf.image.resize_images(pic, size=self.data_dims[0:2], 
                                                                                                                               method=tf.image.ResizeMethod.BILINEAR)),
                                                                    label, *kw), self.num_threads)
        
        elif(self.resize_method == 'QUAD_RESIZING'):
            self.dataset = self.dataset.map(lambda pic, label, *kw : (tf.image.per_image_standardization(
                                                                    tf.image.resize_images(pic, 
                                                                                           size=self.data_dims[0:2], 
                                                                                           method=tf.image.ResizeMethod.BICUBIC)),
                                                                 label, *kw), self.num_threads)
        
        elif(self.resize_method == 'ZERO_PADDING'):
            self.dataset = self.dataset.map(lambda pic, label, *kw : (self._zero_padding(tf.image.per_image_standardization(pic)),
                                                                 label, *kw), self.num_threads)
            
        else:
            print('Error: unknown resizing method')
            raise
    
    
    def create_tf_dataset_and_preprocessing(self, features, labels, metadata=None):
        print('Constructing the new TensorFlow input pipeline on device {}...'.format(self.tf_device))
        with self.graph.as_default(), tf.device(self.tf_device):
            if(metadata is None):
                output_types = (tf.float32, tf.int32)
                output_shapes = (tf.TensorShape([None, None, self.data_dims[2]]), tf.TensorShape([]))
            else:
                output_types = (tf.float32, tf.int32, tf.string)
                output_shapes = (tf.TensorShape([None, None, self.data_dims[2]]), tf.TensorShape([]), tf.TensorShape([]))

            self.dataset = tf.data.Dataset.from_generator(lambda: self.generator(features, labels, metadata),
                                                      output_types = output_types,
                                                      output_shapes = output_shapes)
            if(self.database_name in {'SF', 'MNIST', 'CIFAR-10', 'IMG_NET'}):
                print('Data preprocessing...')
                self.data_preprocessing()
                print('Grouping pictures by size...')
                self.dataset = self.dataset.apply(tf.contrib.data.group_by_window(lambda pic, label, *kw: self._get_key_from_tensor(pic),
                                                                                  lambda key, tensors : tensors.batch(self.batch_size),
                                                                                  window_size=self.batch_size))
            elif(self.database_name == 'SF_LSTM'):
                print('Getting sequence length from input features....')
                self.seq_length = self.dataset.apply(tf.contrib.data.map_and_batch(lambda seq, label, *kw: tf.shape(seq)[0], 
                                                                                   self.batch_size, self.num_threads))
                self.seq_length_iterator = self.seq_length.make_one_shot_iterator()
                print('Zero-padding each sequence according to the max seq length in each batch...')
                self.dataset = self.dataset.padded_batch(self.batch_size, padded_shapes=output_shapes)
            
            self.dataset = self.dataset.prefetch(buffer_size = self.prefetch_buffer_size)
            self.data_iterator = self.dataset.make_one_shot_iterator()
            print('New TF Dataset created.')
    
    def get_next_batch(self):
        if(self.database_name == 'SF_LSTM'):
            return self.data_iterator.get_next(), self.seq_length_iterator.get_next()
        else:
            return self.data_iterator.get_next()
    
    
    

                
