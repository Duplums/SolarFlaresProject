
'''
This class aims to create a list of files where we can find the data and to manage
the computer memory. The data memory size (in MB) is given and should not be exceeded.
It also builds the input pipeline for the whole TensorFlow computation.
'''

import os, re, pickle, traceback, math, drms, sys
from collections import OrderedDict
import skimage.transform as sk
import tensorflow as tf
import numpy as np
import h5py as h5
    
class Data_Gen:
    
    main_path = None
    paths_to_file = None # array of paths where we can find the data
    nb_total_files = None
    num_files_analyzed = None
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
    model_name = None
    time_step = None
    nb_classes = None
    subsampling = None
    resize_method = None
    dataset = None
    data_iterator = None
    seq_length = None
    features = None
    labels = None
    metadata = None
    seq_length_iterator = None
    training_mode = None
    pb_kind = None
    flare_level = None
    
    def __init__(self, data_name, config, training=True, max_pic_size=None, verbose = False):
        assert data_name in {'SF', 'SF_encoded', 'MNIST', 'CIFAR-10', 'IMG_NET'}
 
        self.features = []
        self.labels = []
        self.metadata = []
        self.memory_size =  config['batch_memsize']
        self.data_dims = config['data_dims']
        self.database_name = data_name
        self.pb_kind = config['pb_kind']
        self.training_mode = training
        self.model_name = config['model']
        self.nb_classes = config['nb_classes']
        self.batch_size = config['batch_size']
        self.num_threads = config['num_threads']
        self.prefetch_buffer_size = config['prefetch_buffer_size']
        self.max_pic_size = max_pic_size
        self.subsampling = 1
        
        if(data_name in {'SF', 'SF_encoded'}):
            assert config['resize_method'] in {'NONE', 'LIN_RESIZING', 'QUAD_RESIZING', 'ZERO_PADDING'}
            assert os.path.isdir(config['input_features_dir'])
            assert os.path.isdir(config['output_features_dir'])
            self.segs = config['segs']
            self.flare_level = config['flare_level']
            self.input_features_dir = config['input_features_dir']
            self.output_features_dir = config['output_features_dir']
            self.resize_method = config['resize_method']
            self.rescaling_factor = config['rescaling_factor']
            self.time_step = config['time_step']
            self.output_features = {}
            self.output_labels = {}
            if(training):
                self.subsampling = config['subsampling']

        if(training):
            self.main_path = config['training_paths']
        else:   
            self.main_path = config['testing_paths']

        if(data_name in {'MNIST', 'CIFAR-10', 'IMG_NET'}):
            self.resize_method = config['resize_method']
            print('Warning: no preprocessing for this data base. We assert that the HDF5 files are organized as follows:\n')
            print('\t/features[dataset]\n')
            print('\t/labels[dataset]\n')
        
        
        # First checks
        self.init_paths_to_file(verbose)
        
        if(self.max_pic_size is None and data_name == 'SF'):
            print('Computing the maximum of picture size...')
            self.max_pic_size = self.get_max_size()
        if(self.max_pic_size is None or max_pic_size[0] < 0 or self.max_pic_size[1] < 0):
            print('Warning: maximum size unknown ({}), could lead to error'.format(self.max_pic_size))
        else:
            print('Maximum size found : {}'.format(self.max_pic_size))
    
    # Returns a list of all the files inside the directory 
    # (can be recursive search with recursive_search=True)
    @staticmethod
    def file_scanning(path, recursive_search=False, list_files = []):
        if(os.path.isdir(path)):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if(recursive_search):
                    Data_Gen.file_scanning(file_path, recursive_search, list_files)
                elif(os.path.isfile(file_path)):
                    list_files += [file_path]
        elif(os.path.isfile(path)):
            list_files += [path]
        return list_files
    
    def init_paths_to_file(self, verbose = False):
        self.paths_to_file = []
        self.size_of_files = []
        self.nb_total_files = 0
        self.num_files_analyzed = 0
        nb_files_ignored = 0
        for path in Data_Gen.file_scanning(self.main_path, True):
            size = os.path.getsize(path)/(1024*1024)
            if(size <= self.memory_size):
                self.paths_to_file += [path]
                self.size_of_files += [size/float(self.subsampling)]
                self.nb_total_files += 1
            else:
                if(verbose):
                    print('Warning: file {} [{}MB] will not fit in memory (> {}MB). Ignored'.format(os.path.basename(path), size, self.memory_size))
                nb_files_ignored += 1
        print('Number of files ignored (>{}MB): {}'.format(self.memory_size, nb_files_ignored))
    
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
    # time series. If 'time_event_last_frame', then the eruption occurs in the last
    # frame so the result needs to be reversed.
    @staticmethod
    def _extract_timeseries_from_video(vid, scalars, channels, time_event_last_frame = True):
        res = [[] for k in range(len(scalars))]
        sample_time = []
        tf = drms.to_datetime(vid.attrs['end_time'])
        last_frame = None
        for frame_key in sorted(list(vid.keys()), key=lambda frame_key : float(frame_key[5:])):
            if('channels' in vid[frame_key].keys() and
               len(vid[frame_key]['channels'].shape) == 3):
                ti = drms.to_datetime(vid[frame_key].attrs['T_REC'])
                sample_time += [(tf - ti).total_seconds()/60]
                i = 0
                for scalar in scalars:
                    if(scalar == 'l1_err' or scalar == 'TV'):
                        l1_err = 0
                        TV = 0
                        try:
                            this_frame = Data_Gen._extract_frame(vid[frame_key]['channels'], vid[frame_key].attrs['SEGS'], channels)
                            if(last_frame is None):
                                last_frame = this_frame
                            for c in range(last_frame.shape[2]):
                                if(channels is None or vid[frame_key].attrs['SEGS'][c].decode() in channels):
                                        if(scalar == 'l1_err'):
                                            l1_err += np.sum(np.abs(sk.resize(this_frame[:,:,c], last_frame.shape[:2], preserve_range=True)-last_frame[:,:,c]))
                                        else:
                                            # only valid total variation
                                            TV += np.sum(np.sqrt(np.square(np.diff(this_frame[:,:,c], axis=0)[:, 1:])) + 
                                                                 np.square(np.diff(this_frame[:,:,c], axis=1)[1:, :]))
                            if(channels is None):
                                nb_channels = last_frame.shape[2]
                            else:
                                nb_channels = len(channels)
                            normalization = np.product(last_frame.shape[0:2])*nb_channels
                            if(scalar == 'TV'):
                                res[i] += [TV/normalization]
                            else:
                                res[i] += [l1_err/normalization]
                        except:
                            sample_time = sample_time[:-1]
                            print('Frame {} not extracted.'.format(frame_key))
                            print(traceback.format_exc())
                    else:
                        res[i] += [vid[frame_key].attrs[scalar]]
                    i += 1
                last_frame = this_frame

        if(time_event_last_frame):
            return np.flip(np.array(res), axis=1), np.flip(np.array(sample_time), axis=0)
        return np.array(res), np.array(sample_time)
        
    # Extracts some scalars from video that evolve according to the time (ex: SIZE of a frame).
    # The scalar must be in the frame attributes of a video (exception for 'TV' and 'l1_err': they are
    # computed directly in '_extract_timeseries_from_video'). These time series are concatenated
    # in one list for every video. 'tstart' and 'tend' are used to know when the time series
    # begin and end (from an event, time reversed). If values are missing (<5% by default), 
    # they are interpolated.
    
    # NOTE: all frames MUST have 'T_REC' and 'SEGS' in their attribute. All videos MUST have 'end_time'
    # in their attribute.
    @staticmethod
    def extract_timeseries(paths_to_file = [], 
                           scalars = [], 
                           channels = None, 
                           time_step=60, 
                           tstart=0, 
                           tend=60*24, 
                           loss=0.05):
        nb_frames = int((tend - tstart)/time_step) + 1
        nb_scalars = len(scalars)
        sample_time = np.linspace(tstart, tend, nb_frames)
        res = []
        print('{} frames are considered from {}min before a solar eruption to {}min.'.format(nb_frames, tstart, tend))
        if(type(paths_to_file) is not list and os.path.isdir(paths_to_file)):
            print('Path to a directory. All the files and directories inside are scanned')
            paths_to_file = Data_Gen.file_scanning(paths_to_file, recursive_search=True)
        print('INFO: a linear interpolation is used to reconstruct the time series.')
        for file_path in paths_to_file:
            try:
                with h5.File(file_path, 'r') as db:
                    print('Analyzing file {}'.format(os.path.basename(file_path)))
                    for vid_key in db.keys():
                        vid_time_series, vid_sample_time = Data_Gen._extract_timeseries_from_video(db[vid_key], scalars, channels)
                        if(len(vid_sample_time) > 0):
                            i_start = np.argmin(abs(vid_sample_time - tstart))
                            i_end = np.argmin(abs(vid_sample_time - tend))
                            #if(np.any(np.isnan(vid_time_series))):
                            #    print('Video {} ignored because the time series associated contains \'NaN\'.'.format(vid_key))
                            if(abs(vid_sample_time[i_start] - tstart) <= time_step):
                                nb_frames_in_vid = i_end - i_start + 1
                                if(1 - nb_frames_in_vid/nb_frames <= loss):
                                    res_vid = np.zeros((nb_scalars, nb_frames), dtype=np.float32)
                                    for k in range(nb_scalars):
                                        res_vid[k,:] = np.interp(sample_time, vid_sample_time, vid_time_series[k,:])
                                    res += [res_vid]                            
            except:
                print('Impossible to extract time series from file {}'.format(file_path))
                print(traceback.format_exc())
                raise

        return np.array(res, dtype=np.float32), sample_time
    
    # Assigns a label (int number) associated to a flare class.
    # This label depends of the number of classes for a classification pb.
    def _label(self, flare_class):
        if(self.pb_kind in {'classification', 'encoder'}):   
            if(self.nb_classes == 2):
                return int(flare_class[0] >= 'M')
            else:
                print('Number of classes > 2 case : not yet implemented')
                raise
        elif(self.pb_kind == 'regression'):
            return np.log(self.flare_level[flare_class[0]] * float(flare_class[1:]))
        else:
            print('Illegal problem for assigning a label: {}'.format(self.pb_kind))
            raise
    
    # Resizes every frame in a video according to the data_dims param (if None: 
    # size of the first frame, dim H x W x C) (with the resizing method 'self.resize_method'). 
    # Returs a numpy array of size n x H x W x C if n = nb of frames
    def _resize_video(self, video):
        n = len(video)
        if(n == 0):
            return np.array([])
        if(len(video[0].shape) != 3):
            raise RuntimeError('The first frame of a video has the following invalid dimension: {}'.format(video[0].shape))
        # Set the output shape expected
        if(None in self.data_dims[1:]):
            if(self.resize_method == 'ZERO_PADDING'):
                (H, W, C) = np.max([pic.shape for pic in video], axis=0)
            else:
                (H, W, C) = video[0].shape
        else:
            (H, W, C) = self.data_dims[1:4]
            
        output_shape = (int(np.round(H*self.rescaling_factor)), int(np.round(W*self.rescaling_factor)), C)
        resized_vid = np.zeros((n,) + output_shape , dtype=np.float32)
        # Set the order of the interpolation if resize_method in {'NONE', 'LIN', 'QUAD'}
        if(self.resize_method == 'NONE'):
            o = 1
        elif(self.resize_method == 'LIN_RESIZING'):
            o = 1 
        elif(self.resize_method == 'QUAD_RESIZING'):           
            o = 2
        elif(self.resize_method == 'ZERO_PADDING'):
            o = -1
        else:
            raise RuntimeError('Unknown resized method: {}'.format(self.resize_method))
            
        for k in range(n):
            if(o == -1):
                # We are centering the picture and adding constant edge values around
                (Hi, Wi) = video[k].shape[:2]
                eps_w = int(np.round((W-Wi)/2)) # marge width
                eps_h = int(np.round((H-Hi)/2)) # marge height 
                resized_vid[k, :, :, :] = np.pad(video[k], ((eps_h, H-Hi-eps_h), (eps_w, W-Wi-eps_w), (0, 0)), 'edge')
            else:
                resized_vid[k, :, :, :] = sk.resize(video[k], output_shape, order = o, preserve_range=True)
        
        return resized_vid
    
    
    # It orders a list of strings, assuming it has one of the following format:
    #   * frame{} (ex: 'frame0', 'frame98', ...)
    #   * {} (ex: '1', '100', ...)
    @staticmethod
    def _ordered_frames(list_frames):
        if all([re.match('frame[0-9]+', f) for f in list_frames]):
            return sorted(list_frames, key=lambda f: int(f[5:]))
        elif(all([re.match('[0-9]+', f) for f in list_frames])):
            return sorted(list_frames, key=lambda f: int(f))
        else:
            raise RuntimeError('Unknown frame format: {}'.format(list_frames))
        
        
        
    
    # check 'NaN' in a frame that can have one or multiple channels. Returns
    # a clean frame. INPUT: np array with shape (h, w, c)
    @staticmethod
    def _check_nan(frame, verbose = False):
        shape = frame.shape
        new_frame = None
        if(len(shape) != 3):
            print('Shape of frame must be (h, w, c) (got {})'.format(shape))
            raise        
        for c in range(shape[2]):
            pic = frame[:,:,c]
            if(np.any(np.isnan(pic))):
                if(verbose):
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
                    if(verbose):
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
    def _extract_frame(frame, frame_segs, frame_final_segs = None, verbose = False):
        if(any([type(seg) == bytes or type(seg) == np.bytes_ for seg in frame_segs])):
            frame_segs = [seg.decode() for seg in frame_segs]
        if(frame_final_segs is None):
            n = frame.shape[2]
            nb_channels = n
        else:
            n = len(frame_segs)
            nb_channels = len(frame_final_segs)
        shape_frame = frame.shape[0:2] + (nb_channels,)
        frame_tensor = np.zeros(shape_frame, dtype=np.float32)
        channel_counter = 0
        # Considers only certain segments
        for k in range(n):
            if(frame_final_segs is None or 
               (frame_final_segs is not None and 
               frame_segs[k] in frame_final_segs)):
                frame_tensor[:,:,channel_counter] = frame[:,:,k]
                channel_counter += 1
        # Checks 'NaN' (be careful ,the size might change)
        frame_tensor = Data_Gen._check_nan(frame_tensor, verbose)
        return frame_tensor
        
    # Extract the data from the list of files according to the parameters set.
    # OUTPUT : 2 lists that contains pictures of possibly various sizes and 
    # the labels associated. NOTE: if vid_infos is true, another list containing
    # video information relatively to the picture is added. Each meta data is 
    # formatted as : '{label}|{name_file}|{name_vid}|{frame_nb}'
    def _extract_data(self, files_to_extract, 
                      saving = False, 
                      retrieve = False, 
                      vid_infos = False, 
                      resize_pic_in_same_vid = False,
                      verbose = False):
        features = []
        labels = []
        metadata = []
        memory_used = 0
        for file_path in files_to_extract:
            if(os.path.isfile(file_path)):
                try:
                    with h5.File(file_path, 'r') as db:
                        if(verbose):
                            print('Beginning to extract data from {}'.format(os.path.basename(file_path)))
                        if(self.database_name == 'SF' or self.database_name == 'SF_encoded'):
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
                                    memory_used += os.path.getsize(curr_features)
                                    memory_used += os.path.getsize(curr_labels)
                                    if(verbose):
                                        print('Data retrieved from {}, {}'.format(features_name, labels_name))
                                    if(vid_infos and verbose):
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
                                    video = [] 
                                    label = self._label(db[vid_key].attrs['event_class'])
                                    meta = '{}|{}|{}|'.format(db[vid_key].attrs['event_class'], os.path.basename(file_path), vid_key)
                                        
                                    for frame_key in self._ordered_frames(list(db[vid_key].keys())):
                                        # subsample the video
                                        if(frame_counter % self.subsampling == 0):
                                            if('channels' in db[vid_key][frame_key].keys()):
                                                if(self.database_name == 'SF'):
                                                    frame_tensor = Data_Gen._extract_frame(db[vid_key][frame_key]['channels'], db[vid_key][frame_key].attrs['SEGS'], self.segs, verbose)
                                                else:
                                                    frame_tensor = np.array(db[vid_key][frame_key]['channels'])
                                                if(frame_tensor is None):
                                                    if(len(self.segs) == 0):
                                                        print('Warning: no segments to extract.')
                                                        self.features.clear()
                                                        self.labels.clear()
                                                        self.metadata.clear()
                                                        return
                                                    else:
                                                        raise RuntimeError('None frame in file {}, video {}'.format(file_path, vid_key))
                                                if(self.model_name == 'LRCN'):
                                                    video += [frame_tensor]
                                                else:
                                                    if(not resize_pic_in_same_vid):
                                                        curr_features += [frame_tensor]
                                                    else:
                                                        video += [frame_tensor]
                                                    curr_labels += [label]
                                                    curr_meta += [meta+frame_key]
                                                
                                                memory_used += frame_tensor.nbytes
                                    # 1 sample = 1 video
                                    if(self.model_name == 'LRCN' and len(video) > 0):
                                        curr_features += [self._resize_video(video)]
                                        curr_labels += [label]
                                        curr_meta += [meta]
                                    # expands the video
                                    elif(resize_pic_in_same_vid and len(video) > 0):
                                        curr_features += self._resize_video(video).tolist()
                                    
                                print('Data extracted from {}.'.format(os.path.basename(file_path)))
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
                                curr_features = np.array(db['features']).tolist()
                                curr_labels = np.array(db['labels']).tolist()
                                memory_used += np.array(db['features']).nbytes + np.array(db['labels']).nbytes
                            
                            elif(self.database_name == 'SF_LSTM'):
                                print('Not implemented yet !')
                                raise
                            
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
        # Clears properly the arrays
        self.features.clear()
        self.labels.clear()
        self.metadata.clear()
        
        # Stores the data in memory 
        self.features = features
        self.labels = labels
        if(vid_infos):
            self.metadata= metadata
        if(memory_used > 0):
            print('Memory used: {}MB'.format(memory_used/(1024*1024)))
        return (len(self.features) == 0)
    
    def gen_batch_dataset(self, 
                   save_extracted_data = False, 
                   retrieve_data = False,
                   take_random_files = False,
                   get_metadata = False,
                   resize_pic_in_same_vid = False,
                   rm_paths_to_file = True,
                   verbose = False):
        
        batch_mem = 0
        if(take_random_files):
            files_index = np.random.permutation(len(self.size_of_files))
        else:
            # Heuristic to know which files we will consider
            files_index = np.argsort(self.size_of_files)
        
        # Stores the file paths according to their size (by taking random files or 
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
        self.num_files_analyzed += counter
        if(rm_paths_to_file):
            self.size_of_files = [s for k, s in enumerate(self.size_of_files) if k not in files_index[:counter]]
            self.paths_to_file = [s for k, s in enumerate(self.paths_to_file) if k not in files_index[:counter]]
        
        if(verbose):
            print('Files to be loaded in memory : ')
            for f in files_in_batch:
                print('\t - {} => {}MB'.format(os.path.basename(f), math.ceil(os.path.getsize(f)/(1024.0*1024))))
        
        # Loads the data in memory
        end_of_data = self._extract_data(files_in_batch, 
                                         save_extracted_data, 
                                         retrieve_data, 
                                         vid_infos = get_metadata, 
                                         resize_pic_in_same_vid = resize_pic_in_same_vid,
                                         verbose=verbose)
        if(len(self.features) > 0):
            print('{} elements extracted.\n'.format(len(self.features)))
        return(end_of_data)
    
    def get_num_files_analyzed(self):
        return self.num_files_analyzed
    
    def get_num_total_files(self):
        return self.nb_total_files
    
    def get_num_features(self):
        return len(self.features)
    
    # Adds the features extracted by the Neural Network to the 'output_features' dictionnary.
    def add_output_features(self, features, metadata):
        assert len(features) == len(metadata)
        
        for k in range(len(metadata)):
            meta = metadata[k].decode()
            # meta == 'label|file|video|frame_nb'
            if(re.match('.+\|.+\|.+\|.+', meta)):
                meta_list = re.split(r'\|', meta)
                if(len(meta_list) == 4):
                    label = meta_list[0]
                    file = meta_list[1]
                    video = meta_list[2]
                    frame_nb = meta_list[3]
                    if(file in self.output_features and video in self.output_features[file]):
                        if(frame_nb not in self.output_features[file][video]):
                            self.output_features[file][video].update({frame_nb: features[k]})
                        else:
                            print('Warning: frame {} already exists in file {}, video {}. Ignored'.format(frame_nb, file, video))
                    elif(file in self.output_features):
                        self.output_features[file].update({video: {frame_nb: features[k]}})
                        self.output_labels[file].update({video: label})
                    else:
                        self.output_features.update({file: {video: {frame_nb: features[k]}}})
                        self.output_labels.update({file: {video: label}})
                else:
                    print('Impossible to parse the metadata {}.Feature ignored'.format(meta))
            else:
                print('Unknown format for metadata {}. Feature ignored.'.format(meta))
    
    def dump_output_features(self):
        for f_key, f_val in self.output_features.items():
            file_path = os.path.join(self.output_features_dir, f_key)
            with h5.File(file_path, 'w') as f:
                for vid_key, vid_val in f_val.items():
                    f.create_group(vid_key)
                    for frame_key, frame_val in vid_val.items():
                        f[vid_key].create_group(frame_key)
                        f[vid_key][frame_key].create_dataset('channels', data=np.array(frame_val, dtype=np.float32))
                    f[vid_key].attrs['event_class'] = self.output_labels[f_key][vid_key]
            print('File {} saved.'.format(f_key))
        
        # Clears the dictionnaries and increments the part counter
        self.output_features.clear()
        self.output_labels.clear()

     # Used as input for the TensorFlow pipeline
    @staticmethod
    def generator(features, labels, metadata, use_metadata = False):
        if(not use_metadata):
            for k in range(len(features)):
                yield (features[k], labels[k])
        else:
            for k in range(len(features)):
                yield (features[k], labels[k], metadata[k])
        
    #############################################################
    #                   TENSORFLOW GRAPH                        #
    #############################################################
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
            raise RuntimeError('Error: unknown resizing method')
    
    def create_tf_dataset_and_preprocessing(self, use_metadata = False):
        if(self.pb_kind in {'classification', 'encoder'}):
            output_types = (tf.float32, tf.int32)
        elif(self.pb_kind == 'regression'):
            output_types = (tf.float32, tf.float32)
        else:
            print('Illegal kind of problem: {}'.format(self.pb_kind))
            raise
        output_shapes = (tf.TensorShape([None for k in range(len(self.data_dims)-1)] + [self.data_dims[-1]]), tf.TensorShape([]))
        
        if(use_metadata):
            output_types += (tf.string,)
            output_shapes += (tf.TensorShape([]),)
        
        self.dataset = tf.data.Dataset.from_generator(lambda: Data_Gen.generator(self.features, self.labels, self.metadata, use_metadata),
                                                      output_types = output_types,
                                                      output_shapes = output_shapes)
        
        if(self.model_name in {'VGG_16', 'VGG_16_encoder_decoder'}):
            self.data_preprocessing()
            self.dataset = self.dataset.apply(tf.contrib.data.group_by_window(lambda pic, label, *kw: self._get_key_from_tensor(pic),
                                                                              lambda key, tensors : tensors.batch(self.batch_size),
                                                                              window_size=self.batch_size))
            self.dataset = self.dataset.prefetch(buffer_size = self.prefetch_buffer_size)
            
        elif(self.model_name == 'LSTM'):
            self.seq_length = self.dataset.apply(tf.contrib.data.map_and_batch(lambda seq, label, *kw: tf.shape(seq)[0], 
                                                                               self.batch_size, self.num_threads))
            self.seq_length_iterator = self.seq_length.make_initializable_iterator()
            self.dataset = self.dataset.padded_batch(self.batch_size, padded_shapes=output_shapes)
            self.dataset = self.dataset.prefetch(buffer_size = self.prefetch_buffer_size)
        
        elif(self.model_name == 'LRCN'):
            self.dataset = self.dataset.map(lambda video, label, *kw: (tf.map_fn(lambda img : tf.image.per_image_standardization(img), video, parallel_iterations=self.num_threads),
                                                                       label, *kw),
                                            self.num_threads)
            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.prefetch(buffer_size = self.prefetch_buffer_size)
        
        self.data_iterator = self.dataset.make_initializable_iterator()
        
    def get_next_batch(self):
        if(self.database_name == 'SF_LSTM'):
            return (self.data_iterator.get_next(), self.seq_length_iterator.get_next())
        else:
            return self.data_iterator.get_next()
    
    
    

                
