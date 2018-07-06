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

# Get every file path contained in path
def get_files_path(path, regex_name = '*'):
     os.chdir(path)
     return [path+'/'+file_name for file_name in glob.glob(regex_name)]

# Tranforms a binary file into a Python object
def get_db(file_path):
    with open(file_path, 'rb') as f:
        db = pickle.load(f)
    return db

def get_pictures_size(positive_path, negative_path, show=False):
    files_positive = get_files_path(positive_path)
    files_negative = get_files_path(negative_path)
    pictures_size = []
    for f in files_negative:
        db = get_db(f)
        for k, pic in db.items():
            pictures_size += [list(np.shape(pic['Br']))]
    for f in files_positive:
        db = get_db(f)
        for k, pic in db.items():
            pictures_size += [list(np.shape(pic['Br']))]
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
    
    return np.array(pictures_size)

# STEP 1: Resize the data according to 'resized_pic_shape' param
# STEP 2: Normalize the data
# STEP 3: Return a new Tensor dataset containing only data (no metadata)
def preprocess_db(db, resized_pic_shape, segs, method='linear_interp'):
    assert method in ['linear_interp', 'zero_padding']
    
    nb_channels = len(segs)
    nb_pictures = len(db.keys())
    # Memory size for features (in MB)
    mem_size = nb_channels * nb_pictures * \
               resized_pic_shape[0] * resized_pic_shape[1] * 4.0/(1024*1024)
    print('Allocation of memory for features: '+str(mem_size)+ 'MB')
    features = np.zeros((nb_pictures,)+resized_pic_shape+(nb_channels,), dtype = np.float32)
    labels = np.zeros(nb_pictures, dtype = np.bool)
    print('Starting extraction of features and labels...')
    count = 0

    for k, pic in db.items():
        tensor = np.zeros(resized_pic_shape+(nb_channels,), dtype=np.float32)
        for i in range(nb_channels):
            pic_channel = np.array(pic[segs[i]], dtype=np.float32)
            print(pic_channel.shape)
            # check the consistency of the size according to the method used
            if(method == 'zero_padding'):
                if(pic_channel.shape[0] > resized_pic_shape[0] or \
                   pic_channel.shape[1] > resized_pic_shape[1]):
                    warnings.warn('Wrong picture size for zero-padding. Pic size : {} , Max size: {}.\n Picture ignored'.format(pic_channel.shape, resized_pic_shape))
                    break
                resized_pic = np.zeros(resized_pic_shape, dtype=np.float32)
                h_marge = math.floor((resized_pic_shape[0]-pic_channel.shape[0])/2.0)
                w_marge = math.floor((resized_pic_shape[1]-pic_channel.shape[1])/2.0)
                resized_pic[h_marge:h_marge+pic_channel.shape[0],w_marge:w_marge+pic_channel.shape[1]] = pic_channel
            else:
                resized_pic = resize(pic_channel, resized_pic_shape)
            tensor[:,:,i] = normalized(resized_pic)
        features[count] = tensor
        # WE ASSUME THERE'S ONLY ONE FLARE EVENT
        labels[count] =  (pic['flare_events'][0]['event_class'] >= 'M')
        count += 1
        print('Picture '+str(count)+ '/'+str(nb_pictures)+' treated')

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
    if(picture_shape is None and paths_to_file != None):
        try:
            pics_size = get_pictures_size(paths)
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
            resized_pic_shape = (h, w)
            print(' Shape of pictures : '+str(resized_pic_shape))
        except:
            print('Impossible to determine the picture\'s resizing')
            print(sys.exc_info()[0])
            raise
    
    nb_channels = len(segs)
    features = np.zeros((0,)+picture_shape+(nb_channels,), dtype=np.float32)
    labels = np.zeros(0, dtype=np.float32)
    for path in paths:
            os.chdir(path)
            files = glob.glob(regex_name)
            for file in files:
                db = get_db(file)
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


