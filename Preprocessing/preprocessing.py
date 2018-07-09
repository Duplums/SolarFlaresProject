import os, glob, sys, re
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import skimage.transform as sk
import drms
import warnings
# Normalize a picture
def normalized(pic):
    m = np.mean(pic)
    sd = np.std(pic)
    return (pic - m)/sd

# Resize a picture using bicubic interpolation
def resize(pic, shape):
    return sk.resize(pic, shape, order=2, preserve_range=True)

def zero_padding(pic, shape):
    pic_shape = pic.shape
    resized_pic = np.zeros(shape, dtype=np.float32)
    h_marge = math.floor((shape[0]-pic_shape[0])/2.0)
    w_marge = math.floor((shape[1]-pic_shape[1])/2.0)
    resized_pic[h_marge:h_marge+pic_shape[0],w_marge:w_marge+pic_shape[1]] = pic
    return resized_pic

# Get every file path contained in path
def get_files_path(path, regex_name = '*'):
     os.chdir(path)
     return [path+'/'+file_name for file_name in glob.glob(regex_name)]

# Tranforms a binary file into a Python object
def get_db(file_path):
    try:
        with open(file_path, 'rb') as f:
            db = pickle.load(f)
        return db
    except:
        print('Impossible to load data base {}'.format(file_path))
        return None

# We assume that each db is a dict indexed by eruption time and
# which contains videos 
def get_pictures_size(paths_to_file, show=False):
    pictures_size = []
    for f in paths_to_file:
        db = get_db(f)
        if(db != None):
            for _, vid in db.items():
                pictures_size += vid['frames_size']
    if(show):
        # Get an histogram of pictures size.
        plt.hist(pictures_size[:,0])
        plt.xlabel('Height')
        plt.ylabel('Nb of pictures')
        plt.show()
        plt.hist(pictures_size[:,1])
        plt.xlabel('Width')
        plt.ylabel('Nb of pictures')
        plt.show()
    
    return np.array(pictures_size, dtype=np.int32)

# STEP 1: Resize the data according to 'resized_pic_shape' param
# STEP 2: Normalize the data
# STEP 3: Return a new Tensor dataset containing only data (no metadata)
def preprocess_db(db, resized_pic_shape, segs, method='linear_interp'):
    assert method in ['linear_interp', 'zero_padding']
    assert type(db) is dict
    
    nb_channels = len(segs)
    features = np.zeros((0,)+resized_pic_shape+(nb_channels,), dtype = np.float32)
    labels = np.zeros(0, dtype = np.int32)
    print('Starting extraction of features and labels...')

    for erupt_time, video in db.items():
        if(set(video.keys()) >= set(['frames', 'frames_size', 'flare_event'])):
                frames = db['frames']
                frames_size = db['frames_size']
                flare_event = db['flare_event']
                if('event_class' in flare_event.keys()):
                    vid_label = (flare_event['event_class'] >= 'M')
                    if(len(frames_size) == len(frames)):
                        frame_counter = 0
                        for frame in frames:
                            if('channels' in frame.keys()):
                                channels = frame['channels']
                                size = frames_size[frame_counter] # current size of frame
                                if(set(segs) <= set(channels.keys())):
                                    frame_tensor = np.zeros(resized_pic_shape+(nb_channels,), dtype=np.float32)
                                    if(method == 'zero_padding'):
                                        h_marge = math.floor((resized_pic_shape[0] - size[0])/2.0)
                                        w_marge = math.floor((resized_pic_shape[1] - size[1])/2.0)
                                        if(h_marge >= 0 and w_marge >= 0):
                                            for channel_counter in range(nb_channels):
                                                frame_tensor[:,:,channel_counter] = normalized(zero_padding(channels[segs[channel_counter]], resized_pic_shape))
                                            features = np.concatenate((features, frame_tensor))
                                            labels = np.concatenate((labels, vid_label))
                                        else:
                                            print('Wrong frame format for zero-padding (got size {}, expected <= {}) on video {}, frame {}. Ignored'\
                                                  .format(size, resized_pic_shape, erupt_time, frame_counter))
                                    else:
                                        for channel_counter in range(nb_channels):
                                            frame_tensor[:,:,channel_counter] = normalized(resize(channels[segs[channel_counter]], resized_pic_shape))
                                        features = np.concatenate((features, frame_tensor))
                                        labels = np.concatenate((labels, vid_label))
                                else:
                                    print('Wrong channels for video {}, picture {}. Ignored.'.format(erupt_time, frame_counter))
                                
                            else:
                                print('Wrong picture format for video {}, picture {}. Ignored.'.format(erupt_time, frame_counter))
                    else:
                        print('Unable to determine frames size for video {}. Ignored'.format(erupt_time))
                else:
                    print('Unable to determine the label for video {}. Ignored.'.format(erupt_time))
        else:
            print('Wrong video format for eruption {}. Ignored.'.format(erupt_time))
            
    return (features, labels)

# 'rescale_method' can whether be 'max' or 'median'
def create_tf_dataset(paths_to_file, picture_shape = None, regex_name = '*',\
                      segs= ['Bp', 'Bp_err', 'Br', 'Br_err', 'Bt', 'Bt_err', 'Dopplergram',\
                             'continuum', 'magnetogram'],\
                      rescale_method='max'):
    assert rescale_method in ['max', 'median']
    
    if(rescale_method == 'max'):
        preprocessing = 'zero_padding'
    else:
        preprocessing = 'linear_interp'
    if(picture_shape is None):
        try:
            pics_size = get_pictures_size(paths_to_file)
            if(rescale_method == 'median'):
                # Take the median size 
                medHeight = np.median(pics_size[:,0])
                medWidth = np.median(pics_size[:,1])
                h = 2**int(np.log(medHeight)/np.log(2))
                w = 2**int(np.log(medWidth)/np.log(2))
            elif(rescale_method == 'max'):
                # Take the max size 
                maxHeight = max(pics_size[:,0])
                maxWidth = max(pics_size[:,1])
                h = 2**math.ceil(np.log(maxHeight)/np.log(2)) # round up
                w = 2**math.ceil(np.log(maxWidth)/np.log(2))  # round up
                return (h, w)
            picture_shape = (h, w)
            print(' Shape of pictures : '+str(picture_shape))
        except:
            print('Impossible to determine the picture\'s resizing')
            print(sys.exc_info()[0])
            raise
    
    nb_channels = len(segs)
    features = np.zeros((0,)+picture_shape+(nb_channels,), dtype=np.float32)
    labels = np.zeros(0, dtype=np.float32)
    for path in paths_to_file:
        db = get_db(path)
        f, l = preprocess_db(db, picture_shape, segs, preprocessing)
        features = np.concatenate((features, f))
        labels = np.concatenate((labels, l))
        mem_size = (features.nbytes + labels.nbytes)/(1024*1024)
        print('Total memory size : '+str(mem_size)+ ' MB')
    return (features, labels)

# Returns n timeseries where n = nb of features for 1 video
# and the sample time from the event (in minutes)
    # res : matrix of size n x p where p = nb of frames, n = nb of features
def get_timeseries_from_video(vid, features = ['SIZE', 'SIZE_ACR', 'NACR']):
    res = np.zeros((len(features), len(vid['frames'])), dtype=np.float32)
    sample_time = np.zeros(len(vid['frames']), dtype=np.float32)
    tf = drms.to_datetime(vid['flare_event']['end_time'])
    i = 0
    j = 0
    for pic in vid['frames']:
        ti = drms.to_datetime(pic['header']['T_REC'])
        sample_time[j] = (tf - ti).total_seconds()/60
        for feature in features:
            res[i,j] = pic['header'][feature]
            if(np.isnan(res[i,j])):
                warnings.warn('Timeseries \'{}\' contains \'Nan\' for video {}'.format(feature, ti))
            i += 1
        j += 1
        i = 0
    return res, sample_time


def plot_timeseries_from_video(vid, features = ['SIZE', 'SIZE_ACR', 'NACR']):
    timeseries, sample_time = get_timeseries_from_video(vid, features)
    (n,p) = np.shape(timeseries)
    fig, axes = plt.subplots(n, 1)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.plot(sample_time, timeseries[i,:])
        ax.set_ylabel(features[i])
        ax.set_xlabel('Time before event (in minutes)\n Flare class {}'.format(vid['flare_event']['event_class']))
    plt.show()


# Some time series contains missing values. Only the ones starting at tstart 
# and containing < loss missing values are taken into account. Linear interp
# is used.
def create_timeseries_dataset(db,  features = ['SIZE', 'SIZE_ACR', 'NACR'], \
                              tstart = 12, tend = 90*12, loss=0.05):
    assert 0 <= loss < 1 and tstart <= tend <= 120*12 # max: 24h
    timestep = 12 # min
    nb_frames = int((tend - tstart)/timestep) + 1
    nb_ts = len(features)
    sample_time_th = np.linspace(tstart, tend, nb_frames)
    # we do not know yet how many videos match
    res = []
    for key, vid in db.items():
        ts, sample_time = get_timeseries_from_video(vid)
        sample_time = np.array(sample_time, dtype = np.float32)
        i_start = np.argmin(abs(sample_time - tstart))
        i_end = np.argmin(abs(sample_time - tend))
        if(np.any(np.isnan(ts))):
            warnings.warn('Video {} ignored because it contains \'NaN\'.'.format(key))
        elif(abs(sample_time[i_start] - tstart) <= timestep):
            nb_frames_in_vid = i_end - i_start + 1
            if(1 - nb_frames_in_vid/nb_frames <= loss):
                res_vid = np.zeros((nb_ts, nb_frames), dtype=np.float32)
                for k in range(nb_ts):
                    res_vid[k,:] = np.interp(sample_time_th, sample_time, ts[k,:])
                res += [res_vid]
    return np.array(res, dtype=np.float32)

def get_timeseries(path, name):
    try:
        ts = pickle.load(open(os.path.join(path, name), 'rb'))
        return ts
    except:
        print('Impossible to restore the time series.')
        return None

def save_tf_dataset(path, features, labels, features_name='features', labels_name='labels'):
    with open(os.path.join(path, features_name), 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(path, labels_name), 'wb') as f:
        pickle.dump(labels, f)

def get_tf_dataset(path, features_name='features', labels_name='labels'):
    try:
        features = pickle.load(open(os.path.join(path, features_name), 'rb'))
        labels = pickle.load(open(os.path.join(path, labels_name), 'rb'))
        return (features, labels)
    except:
        print('Impossible to restore tf_dataset.')
        return (None, None)

def plot_pictures(pics, labels, preds = None, segs='Bp,Br,Bt,Dopplergram,Continuum,Magnetogram'):
    assert len(pics) == len(labels) > 0 and ((preds is None) or len(preds) == len(labels))
    segs = re.split(' ?, ?', segs)
    pics_shape = np.shape(pics)
    if(len(pics_shape) != 4):
        print('Shape of pictures must be nb_pics x height x weight x nb_channels')
        return False
    nb_channels = pics_shape[3]
    nb_pics = pics_shape[0]
    assert len(segs) == nb_channels

    for k in range(nb_pics):
        # Print every channels of an image
        pic = pics[k]
        if(nb_channels == 1):
            plt.imshow(pic[:,:,0])
            if(preds is None):
                plt.xlabel('Label {0}'.format(labels[k]))
            else:
                plt.xlabel('Label {0}, Pred {1}'.format(preds[k]))
            plt.xticks([])
            plt.yticks([])
        else:
            fig, axes = plt.subplots(int(nb_channels/3)+(nb_channels%3>0), \
                                     int(3*(nb_channels>=3)+nb_channels*(nb_channels<3)),\
                                     figsize=(15, 7.5))
            for i, ax in enumerate(axes.flat):
                if(i < nb_channels):
                    # Plot image.
                    ax.imshow(pic[:,:,i])
                    if(preds is None):
                        ax.set_xlabel('{0}, label {1}'.format(segs[i], labels[k]))
                    else:
                        ax.set_xlabel('{0}, \n Label {1}, pred {2}'.format(segs[i], labels[k], preds[k]))
                    # Remove ticks from the plot.
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.subplots_adjust(hspace=0.05, wspace=0.1)

        plt.show()


