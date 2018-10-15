from astropy.io import fits
from sunpy.time import TimeRange
import sunpy.instr.goes as goes_db
from datetime import timedelta
import drms, h5py, cv2, math
import os, csv, traceback, re, glob, sys
import matplotlib.pyplot as plt
import skimage.transform as sk
from scipy import stats
sys.path.append('/home6/bdufumie/SolarFlaresProject')
from CNN import utils
import numpy as np

''' This class aims to download the data from JSOC and to convert it into 
    HDF5 files. For the label and other metadata information, it will be
    stored in the HDF5 files. Each HDF5 file is constructed as follows:
        /video1
            /attrs
                # GOES attributes associated to the event that the video captured
                'event_class': 'M1'
                'event_date': '2010-05-02 00:00:00'
                ...
            /frame1
                /attrs
                    # Active region attributes associated to this frame
                    'SIZE': [100, 200]
                    'T_REC': '2010-05-01 00:00:00'
                    'SEGS': '['Br', 'Bp', 'Bt'] 
                    ...
                /datasets
                    # dtype = int32
                    # shape = (100, 200, 3)
            ...
            /frameM
                
        /video2
        ...
        /videoN
'''

class Data_Downloader:
    # Root path for every files downloaded
    main_path = None
    # Attributes dowloaded in the GOES data base
    goes_attrs = None
    # Attributes downloaded in the JSOC data base
    ar_attrs = None
    # Segments download in the JSOC data base
    ar_segs = None
    # Memory limit for each HDF5 file (in MB)
    mem_limit = None 
    
    def __init__(self, main_path, goes_attrs, ar_attrs, ar_segs, mem_limit = 1024):
        self.main_path = main_path
        self.goes_attrs = goes_attrs
        self.ar_attrs = ar_attrs
        self.ar_segs = ar_segs
        self.mem_limit = mem_limit
        
        if(not os.path.isdir(main_path)):
            os.mkdir(main_path)
        os.chdir(main_path)
    # Downloads the attributes from GOES data base from tstart to tend and 
    # store it into a CSV file.
    def download_goes_data(self, file_name = 'GOES_data.csv',
                           tstart = '2010-05-01', tend = '2018-07-01'):
        if('noaa_active_region' not in self.goes_attrs):
            self.goes_attrs += ['noaa_active_region']
            print('Warning: \'noaa_active_region\' not in goes_attrs. Added.')
        file_path = os.path.join(self.main_path, file_name)
        if(os.path.exists(file_path)):
            print('Be careful, file {} already exists. It will be replaced.'.format(file_path))
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(self.goes_attrs)
            data = goes_db.get_goes_event_list(TimeRange(tstart, tend))
            for row in data:
                if(row['noaa_active_region'] > 0):
                    writing_row = []
                    for attrs in self.goes_attrs:
                        writing_row += [row[attrs]]
                    writer.writerow(writing_row)
    
    # This function aims to select 'relevant' B-class flare events from the
    # GOES csv file. We assume (NOT checked) that this .csv is formated as follows:
    # [class NOAA_ar_num event_date start_time end_time peak_time]
    # The algorithm extracts the B-flares :
    #   - that are not 'too closed' from a M-X flare eruption (for the same AR)
    #   - that are not 'too closed' from a B-flare eruption (for the same AR)
    # 'Too closed' is controlled by the parameter 'time_window' (in days)
    @staticmethod
    def extract_B_flares_from_goes(goes_data_path, output_path, time_window):
        with open(goes_data_path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=',')
            counter_init_B = 0
            counter_final_B = 0
            counter_M_X = 0
            dict_M_X = {}
            dict_B = {}
            # first, store the M-class flares in a dictionnary
            [date, noaa] = [2, 1] # we assume this format
            for event in reader:
                if(re.match('(M|X)[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', str.join(',', event))):
                    counter_M_X += 1
                    event_date = drms.to_datetime(event[date])
                    if(event_date in dict_M_X):
                        dict_M_X[event_date] += [event[noaa]]
                    else:
                        dict_M_X[event_date] = [event[noaa]]
            file.seek(0)
            # Then analyze all B-class flares and store them
            with open(output_path, 'w', newline='') as out:
                writer = csv.writer(out, delimiter=',')
                for event in reader:
                    if(re.match('B[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', str.join(',', event))):
                        counter_init_B += 1
                        event_date = drms.to_datetime(event[date])
                        exclude_event = False
                        for time_delta in range(-time_window, time_window+1):
                            event_date_window  = event_date + timedelta(days=time_delta)
                            if(event_date_window in dict_M_X
                               and event[noaa] in dict_M_X[event_date_window]):
                                #print('Event {} ignored because of M-X flare {}'.format(event,event_date_window))
                                exclude_event = True
                                break
                            elif(event_date_window in dict_B 
                                 and event[noaa] in dict_B[event_date_window]):
                                #print('Event {} ignored because of B flare {}'.format(event, event_date_window))
                                exclude_event = True
                                break
                        if(not exclude_event):
                            writer.writerow(event)
                            counter_final_B += 1
                            if(event_date in dict_B):
                                dict_B[event_date] += [event[noaa]]
                            else:
                                dict_B[event_date] = [event[noaa]]
                    elif(not re.match('(B|C|M|X)[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', str.join(',', event))):
                        writer.writerow(event)
                print('Total number of M-X flares: {}'.format(counter_M_X))
                print('Total number of B flares: {}'.format(counter_init_B))
                print('Number of output B flares: {}'.format(counter_final_B))
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
             
    
    # For all files found, counts the number of videos, frames and channels/frames
    # as well as the mean size and diff between max and min size for each video.
    # Gives also the statistics about the l1-error (RMS) between the first and last
    # frame in each video (with more than 'nb_min_frame_for_rms' frame).
    @staticmethod
    def check_statistics(path_to_files, out = None, nb_min_frame_for_rms=10):
        if(os.path.isdir(path_to_files)):
            files = sorted(glob.glob(os.path.join(path_to_files, '*')))
        elif(os.path.isfile(path_to_files)):
            files = [path_to_files]
        else:
            print('{} is neither a directory nor a file.'.format(path_to_files))
            raise
        close_flag = False
        if(out is None):
            out = sys.stdout
        else:
            try:
                out = open(out, 'w')
                close_flag = True
            except:
                print('Impossible to redirect output to {}. Redirecting to stdout instead'.format(out))
                out = sys.stdout
        results = {'nb_frames' : [], 'avg_size': [], 'nb_channels' : [], 
                   'min_max_size': [], 'rms':[]}
        glob_counter= 0
        for file in files:
            try:
                with h5py.File(file, 'r') as db:
                    for vid_key in db.keys():
                        if(len(db[vid_key].keys())>0):
                            min_size = math.inf
                            max_size = -math.inf
                            avg_size = np.array([0, 0])
                            nb_channels = 0
                            nb_frames = 0
                            l1_err = 0
                            vid_init = False
                            first_frame = None
                            last_frame= None
                            for frame_key in db[vid_key].keys():
                                if('channels' in db[vid_key][frame_key].keys() and
                                    len(db[vid_key][frame_key]['channels'].shape) == 3):
                                    if(not vid_init):
                                        first_frame = Data_Downloader._check_nan(db[vid_key][frame_key]['channels'])
                                        vid_init = True
                                    # !!TO BE CHANGED (ASSUME THAT FRAME_KEY == FRAME[0-..]) !!
                                    if(frame_key == sorted(list(db[vid_key].keys()), 
                                                           key=lambda frame_key : float(frame_key[5:]))[-1]):
                                        last_frame = Data_Downloader._check_nan(db[vid_key][frame_key]['channels'])
                                    max_size = max(max_size, )
                                    min_size = min(min_size, np.prod(db[vid_key][frame_key]['channels'].shape[0:2]))
                                    if(np.prod(db[vid_key][frame_key]['channels'].shape[0:2]) > max_size):
                                        max_size = np.prod(db[vid_key][frame_key]['channels'].shape[0:2])
                                        max_frame_size = db[vid_key][frame_key]['channels'].shape[0:2]
                                    if(np.prod(db[vid_key][frame_key]['channels'].shape[0:2]) < min_size):
                                        min_size = np.prod(db[vid_key][frame_key]['channels'].shape[0:2])
                                        min_frame_size = db[vid_key][frame_key]['channels'].shape[0:2]
                                    nb_channels = db[vid_key][frame_key]['channels'].shape[2]
                                    avg_size += db[vid_key][frame_key]['channels'].shape[0:2]
                                    nb_frames += 1
                            if(nb_frames > nb_min_frame_for_rms):
                                if(first_frame is not None and last_frame is not None):
                                    for c in range(nb_channels):
                                        l1_err += np.sum(np.abs(sk.resize(first_frame[:,:,c], last_frame.shape[:2], preserve_range=True)- last_frame[:,:,c]))
                                    results['rms'] += [l1_err/np.product(last_frame.shape)]
                                else:
                                    print('Unable to find first and last frame in file {}, video {}'.format(file, vid_key))
                                
                                
                            #if(nb_frames > 24):
                            #    print(file+'-'+vid_key+' ('+ db[vid_key].attrs['peak_time']+')'+'=>'+str(nb_frames)+' frames')
                            results['nb_frames'] += [nb_frames]
                            results['avg_size'] += [avg_size/nb_frames]
                            results['min_max_size'] +=[max_size-min_size]
                            results['nb_channels'] += [nb_channels]
                            glob_counter += 1
            except:                
                print('Impossible to get descriptors for file {}'.format(file))
                print(traceback.format_exc())
                break
        out.write('NB OF VIDEOS : {}\n'.format(glob_counter))
        out.write('NB OF FRAMES : {}\n'.format(sum(results['nb_frames'])))
        out.write('MAX FRAME SIZE: {}\n'.format(max_frame_size))
        out.write('MIN FRAME SIZE: {}\n'.format(min_frame_size))
        out.write('NB OF FRAMES / VIDEO:\n')
        out.write('\t'+str(stats.describe(results['nb_frames'])))
        out.write('\nSIZE OF VIDEOS:\n')
        out.write('\t'+str(stats.describe(results['avg_size'])))
        out.write('\nMAX-MIN SIZE OF VIDEOS:\n')
        out.write('\t'+str(stats.describe(results['min_max_size'])))
        out.write('\nNB OF CHANNELS:\n')
        out.write('\t'+str(stats.describe(results['nb_channels'])))
        out.write('\nRMS:\n')
        out.write('\t'+str(stats.describe(results['rms'])))
        if(close_flag):
            out.close()
    
    
    # Display the peak time of each video in each file
    # and redirect the output according to 'out'
    @staticmethod
    def display_peak_time(path_to_files, out = None):
        if(os.path.isdir(path_to_files)):
            files = sorted(glob.glob(os.path.join(path_to_files, '*')))
        elif(os.path.isfile(path_to_files)):
            files = [path_to_files]
        else:
            print('{} is neither a directory nor a file.'.format(path_to_files))
            raise
        close_flag = False
        if(out is None):
            out = sys.stdout
        else:
            try:
                out = open(out, 'w')
                close_flag = True
            except:
                print('Impossible to redirect output to {}. Redirecting to stdout instead'.format(out))
                out = sys.stdout
        for file in files:
            try:
                out.write('File {}:\n'.format(file))
                with h5py.File(file, 'r') as db:
                    for vid_key in db.keys():
                        out.write('\t\'{}\' => {} ({}-flare)\n'.format(vid_key, db[vid_key].attrs['peak_time'], db[vid_key].attrs['event_class']))
            except:                
                print('Impossible to display peak time for file {}'.format(file))
                print(traceback.format_exc())
        if(close_flag):
            out.close()
           
    
    
    # Display a video from .hdf5 file
    @staticmethod
    def display_vid(file, vid, save_pictures=False):
        try:
            with h5py.File(file, 'r') as db:
                video = db[vid]
                frame_keys = list(video.keys())
                if(len(frame_keys) > 0):
                    height, width, nb_channels = video[frame_keys[0]]['channels'].shape
                    channels = video[frame_keys[0]].attrs['SEGS']
                    if(len(channels) != nb_channels):
                        print('Channels supposed to be {} but only {} channels found.'.format(channels, nb_channels))
                        raise
                    for frame_key in frame_keys:
                        channel_count = 0
                        for channel in channels:
                            if(save_pictures):
                                plt.imsave('{}_{}'.format(frame_key, channel) ,arr=video[frame_key]['channels'][:,:,channel_count], cmap='gray')
                            else:
                                cv2.namedWindow(channel.decode(), cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(channel.decode(), height, width)
                                cv2.imshow(channel.decode(), video[frame_key]['channels'][:,:,channel_count])
                            channel_count += 1
                        if(not save_pictures):
                            cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except:
            print('Impossible to display the video.')
            print(traceback.format_exc())
                       
    # Aims to check the integrity of the file 'hdf5_file' according to the 
    # format defined previously. If correct_file is True, then:
    #   * frames with no channels are erased.
    #   * frames with unknown shapes are erased
    #   * if delete_zeros is True, frames that contain channels with only zeros
    #   * are deleted.
    @staticmethod
    def check_integrity(hdf5_file, correct_file = False, delete_zeros = False):
        try:
            with h5py.File(hdf5_file, 'r+') as db:
                print('Analysis of file {} started'.format(hdf5_file))
                nb_vids = len(list(db.keys()))
                if(nb_vids == 0):
                    print('(Warning): 0 video found in the file.')
                report = {}
                for vid_key in db.keys():
                    if(re.match('video[1-9]?[0-9]', vid_key) is None):
                        print('(Warning), video key {} does not match \'video?\'.'.format(vid_key))
                    nb_frames = len(list(db[vid_key]))
                    report.update({vid_key : {}})
                    report[vid_key]['nb_frames'] = nb_frames
                    report[vid_key]['missing_segs'] = []
                    report[vid_key]['incomp_segs'] = []
                    report[vid_key]['no_data'] = []
                    report[vid_key]['NaN'] = []
                    report[vid_key]['zeros'] = []
                    if(nb_frames == 0):
                        print('(Warning): 0 frame found in {}.'.format(vid_key))
                    nb_global_segs = None
                    for frame_key in db[vid_key]:
                        if(re.match('frame[1-9]?[0-9]', frame_key) is None):
                            print('(Warning): frame key {} does not match \'frame?\'.'.format(frame_key))
                        if('SEGS' not in db[vid_key][frame_key].attrs):
                            print('ERROR: attribute \'SEGS\' is missing in frame {}, video {}'.format(frame_key, vid_key))
                            report[vid_key]['missing_segs'] += [frame_key]
                        else:
                            nb_local_segs = len(db[vid_key][frame_key].attrs['SEGS'])
                            if(nb_global_segs is None):
                                nb_global_segs = nb_local_segs
                            elif(nb_global_segs != nb_local_segs):
                                print('WARNING : {} segments found in video {}, frame {} but {} found in the previous frames'.
                                      format(nb_local_segs, vid_key, frame_key, nb_global_segs))
                                report[vid_key]['incomp_segs'] += [frame_key]
                            if('channels' not in db[vid_key][frame_key].keys()):
                                print('ERROR: \'channels\' not found in video {}, frame {}'.format(vid_key, frame_key))
                                report[vid_key]['no_data'] += [frame_key]
                                if(correct_file):
                                    print('--->The frame is erased from video.')
                                    del db[vid_key][frame_key]
                            else:
                                frame_shape = db[vid_key][frame_key]['channels'].shape
                                if(len(frame_shape) != 3):
                                    print('ERROR: Unknown frame shape format : {}'.format(frame_shape))
                                    if(correct_file):
                                        print('--->The frame is erased from video.')
                                        del db[vid_key][frame_key]
                                else:
                                    if(frame_shape[2] != nb_local_segs):
                                        print('WARNING: {} segments found in frame attribute but {} channels found in the data'.format(nb_local_segs, frame_shape[2]))
                                    for k in range(frame_shape[2]):
                                        if(np.any(np.isnan(db[vid_key][frame_key]['channels'][:,:,k]))):
                                            print('(Warning) \'NaN\' found in video {}, frame {}, channel {}'.format(vid_key, frame_key, k))
                                            if(frame_key not in report[vid_key]['NaN']):
                                                report[vid_key]['NaN'] += [frame_key]
                                        if(np.all(db[vid_key][frame_key]['channels'][:,:,k] == 0)):
                                            print('(Info) video {}, frame {}, channel {} contains only zeros'.format(vid_key, frame_key, k))
                                            if(frame_key not in report[vid_key]['zeros']):
                                                report[vid_key]['zeros'] += [frame_key]
                                            if(delete_zeros):
                                                if(correct_file):
                                                    print('--->The frame is erased from video.')
                                                    del db[vid_key][frame_key]
                                                break
                print('Analysis finished with code error 0.')
                print('\n---------FINAL REPORT---------\n')
                for vid_key in db.keys():
                    print('\t\'{}\':\n'.format(vid_key))
                    print('\t\t - {} frames found'.format(report[vid_key]['nb_frames']))
                    print('\t\t - \'SEGS\' attribute missing in frames {}'.format(report[vid_key]['missing_segs']))
                    print('\t\t - incompatible segments between frames {}'.format(report[vid_key]['incomp_segs']))
                    print('\t\t - no data in frames {}'.format(report[vid_key]['no_data']))
                    print('\t\t - \'NaN\' found in frames {}'.format(report[vid_key]['NaN']))
                    print('\t\t - zeros found in frames {}'.format(report[vid_key]['zeros']))
                
        except:
            print('Error while scanning the file.')
            print(traceback.format_exc())
        
        
        
        
    @staticmethod
    def _check_essential_attributes(attrs_set, essential_attrs):
        if(not essential_attrs.issubset(attrs_set)):
            missing_attrs= essential_attrs.difference(attrs_set)
            print('Warning: some attributes are missing to describe an active region: {}. Added.'.format(missing_attrs))
            return list(missing_attrs)
        return []
    
    @staticmethod
    def _in_time_window(time, start_time, end_time):
        if(start_time is None and end_time is None):
            return True
        try:
            if(start_time is None):
                before_end = (drms.to_datetime(time) <= drms.to_datetime(end_time))
                return before_end
            elif(end_time is None):
                after_start = (drms.to_datetime(time) >= drms.to_datetime(start_time))
                return after_start
            else:
                after_start = (drms.to_datetime(time) >= drms.to_datetime(start_time))
                before_end = (drms.to_datetime(time) <= drms.to_datetime(end_time))
                return(after_start and before_end)
        except:
            print('Impossible to determine if time {} is in [{}, {}]'.format(time, start_time, end_time))
            return False
    @staticmethod
    def _UTC2JSOC_time(UTC):
        JSOC = re.sub('-', '.', UTC)
        JSOC = re.sub(' ' , '_', UTC) + '_TAI'
        return JSOC
    
     # Get only the frames that are:
     # * related to our AR (same NOAA)
     # * within +/- 68deg from the central meridian
     # * before the peak time
     # from a list of AR keys.
    @staticmethod
    def _get_frames_key_from_query(ar_nb, peak_time, keys):
        list_keys = []
        for k in range(len(keys.NOAA_AR)):
            if (keys.NOAA_AR[k] == ar_nb and 
                abs(keys.LAT_FWT[k]) <= 68 and 
                abs(keys.LON_FWT[k]) <= 68 and
                drms.to_datetime(keys.T_REC[k]) <= drms.to_datetime(peak_time)):
                list_keys += [k]
 
        return list_keys
    
    # Gets the data from the JSOC data base according to the solar eruptions described
    # in the GOES data base. The queries are based on SunPy and the output files are in 
    # HDF5 format. It returns True if the data has been dowloaded successfully; False otherwise.
    # Args:
    # - files_core_name: each file created will have the following format: {file_core_name}_part_{}.hdf5
    # - directory: each file created will be saved in 'self.main_path/directory'
    # - goes_data_path: path to the GOES.csv file that lists all the flares
    # - goes_row_pattern: regular expression indicating which flares will be downloaded
    # - start_time, end_time: time period considered in the lookup 
    # - nb_frames_before_event: nb of frames downloaded in each video
    # - sample_time: cadence considered for each video (in hours, minimum: 12min <=> 0.2h)
    # - limit: maximum nb of videos (== # of flares) downloaded
    
    def download_jsoc_data(self, files_core_name = 'jsoc_data',
                           directory = None,
                           goes_data_path = None, 
                           goes_row_pattern = '(B|C|M|X)[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', 
                           start_time = None, end_time = None,
                           nb_frames_before_event = 24, 
                           sample_time = 1, # in hours
                           limit = 400):
        
        if(directory is None and not os.path.isdir(os.path.join(self.main_path, 'JSOC-Data'))):
            os.mkdir(os.path.join(self.main_path, 'JSOC-Data'))
            os.chdir(os.path.join(self.main_path, 'JSOC-Data'))
        elif(os.path.isdir(os.path.join(self.main_path, directory))):
            os.chdir(os.path.join(self.main_path, directory))
        else:
            print('The path {} does not exist.'.format(os.path.join(self.main_path, directory)))
            return False
        
        essential_ar_attrs = {'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT', 'T_REC'}
        essential_goes_attrs = {'start_time', 'peak_time', 'noaa_active_region', 'event_class'}
        jsoc_serie = 'hmi.sharp_cea_720s[1-7256]'
        
        # Verifications of the path to GOES data and the format of the .csv
        if(goes_data_path is None):
            if(os.path.exists(os.path.join(self.main_path, 'GOES_data.csv'))):
                goes_data_path = os.path.join(self.main_path, 'GOES_data.csv')
            else:
                print('Please enter a valid path to the GOES data.')
                return False
        if(not os.path.exists(goes_data_path)):
            print('Please enter a valid path to the GOES data.')
            return False
        
        missing_goes_attrs = self._check_essential_attributes(set(self.goes_attrs), essential_goes_attrs)
        if(len(missing_goes_attrs) > 0):
            print('Missing attributes in GOES file : {}.'.format(missing_goes_attrs))
            return False
        
        [start, peak, noaa_ar] = [self.goes_attrs.index('start_time'), 
                                  self.goes_attrs.index('peak_time'),
                                  self.goes_attrs.index('noaa_active_region')]
            
        total_length = sum(1 for line in open(goes_data_path, 'r'))
        self.ar_attrs += self._check_essential_attributes(set(self.ar_attrs), essential_ar_attrs)

        # Estimation of the number of solar eruption videos considered.
        # Limit the number of videos if 'limit' is reached.
        nb_positive = 0
        considered_events = []
        with open(goes_data_path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=',')
            counter = 0
            for event in reader:
                counter += 1
                if(self._in_time_window(event[start], start_time, end_time) and 
                   re.match(goes_row_pattern, str.join(',', event)) and
                   int(event[noaa_ar]) > 0):
                        nb_positive += 1
                        considered_events += [counter]
        if(limit is not None and nb_positive > limit):
            events_really_considered = np.random.choice(considered_events, limit)
            
        # Summary
        print('Nb of videos to download: {}/{}'.format(nb_positive, counter))
        print('Look up of pictures until {}h before an event.'.format(sample_time*nb_frames_before_event))
        

        with open(goes_data_path, 'r', newline='') as file:        
            reader = csv.reader(file, delimiter=',')
            client = drms.Client()
            mem = 0 # Set a counter for the current cache memory (in bytes) used by videos
            part_counter = 0
            vid_counter = 0
            counter = 0
            current_save_file = h5py.File('{}_part_{}.hdf5'.format(files_core_name, part_counter), 'w')
            # Get the delta time for the look up in the JSOC data base (with a marge)
            dt = timedelta(hours=sample_time*(nb_frames_before_event+1))
            # Change the sampling rate format
            sample_time = '@{}h'.format(sample_time)

            for event in reader:
                counter += 1
                if(re.match(goes_row_pattern, str.join(',', event)) and 
                   (limit is None or nb_positive <= limit or
                   (counter in events_really_considered)) and
                    self._in_time_window(event[start], start_time, end_time)):
                    ar_nb = int(event[noaa_ar])
                    # We process only numbered flares
                    if(ar_nb > 0):
                        peak_time = drms.to_datetime(event[peak])
                        start_time = peak_time - dt
                        # Change the date format
                        peak_time = self._UTC2JSOC_time(str(peak_time))
                        start_time = self._UTC2JSOC_time(str(start_time))
                        
                        # Do the request to JSOC database
                        query = '{}[{}-{}{}]'.format(jsoc_serie, start_time, peak_time, sample_time)
                        if(len(self.ar_segs)==0): keys = client.query(query, key=self.ar_attrs)
                        else: keys, segments = client.query(query, key=self.ar_attrs, seg=self.ar_segs)
                        try:
                            # Get only the frames that are:
                            # * related to our AR (same NOAA)
                            # * within +/- 68deg from the central meridian
                            # * before the peak time
                            frames_keys = self._get_frames_key_from_query(ar_nb, peak_time, keys)
                            
                            # Do not download videos with missing data
                            if(len(frames_keys) < nb_frames_before_event):
                                print('Only {} (< {}) frames found for the SF produced on {}. Ignored.'.format(len(frames_keys), nb_frames_before_event, event[peak]))
                            else:
                                current_vid = current_save_file.create_group('video{}'.format(vid_counter))
                                vid_counter += 1
                                for k in range(len(self.goes_attrs)):
                                    current_vid.attrs[self.goes_attrs[k]] = event[k]
                                if(len(frames_keys) > nb_frames_before_event):
                                    print('{} frames are found for the SF produced on {}, only the last {} are considered'.format(len(frames_keys), event[peak], nb_frames_before_event))
                                    frames_keys = frames_keys[len(frames_keys)-nb_frames_before_event:]
                                # We download each video with the LAST frame corresponding to the eruption
                                for i in range(nb_frames_before_event):
                                    current_frame = current_vid.create_group('frame{}'.format(i))
                                    # Includes the specific attributes to the frame
                                    current_frame.attrs['SEGS'] = np.string_(list(self.ar_segs))
                                    for a in self.ar_attrs:
                                        current_frame.attrs[a] = keys[a][frames_keys[i]]
                                        
                                    # Downloads the specific segments
                                    data_frame = []
                                    for seg in self.ar_segs:
                                        url = 'http://jsoc.stanford.edu' + segments[seg][frames_keys[i]]
                                        data = np.array(fits.getdata(url, cache=False), dtype=np.float32)
                                        data_frame += [data]
                                        mem += data.nbytes
                                    data_frame = np.array(data_frame, dtype=np.float32)
                                    # Creates the actual data set in the hdf5 file
                                    current_frame.create_dataset('channels', data=data_frame)

                        except: 
                            print('Impossible to extract data for event {0}.'.format(event[peak]))
                            print(traceback.format_exc())
                else: # if the row pattern does not match 
                    print('Row ignored: '+str.join(',', event))
                
                if(counter % 20 == 0):
                    print('{:0.2f}% of GOES data set analyzed'.format(counter*100.0/total_length))
    
                # Save the current HDF5 file. Reset vid_counter for the next HDF5 file.
                if(mem/(1024*1024) > self.mem_limit):
                    current_save_file.close()
                    part_counter += 1
                    vid_counter = 0
                    mem = 0
                    current_save_file = h5py.File('{}_part_{}.hdf5'.format(files_core_name, part_counter), 'w') 
        
        # After the downloading, close the last file !
        current_save_file.close()
        print('The data base has been downloaded successfully !')
        return True


main_path = '/nobackup/bdufumie/SolarFlaresProject/Data/SF/tmp/'
goes_data_path = '/home6/bdufumie/SolarFlaresProject/DataQuery/GOES_dataset.csv'
goes_attrs = utils.config['SF']['goes_attrs']
ar_attrs = utils.config['SF']['ar_attrs']
ar_segs = utils.config['SF']['segs']

downloader = Data_Downloader(main_path, goes_attrs, ar_attrs, ar_segs)
downloader.download_jsoc_data(files_core_name = 'M_X_jsoc_data',
                           directory = 'M-X-class-flares',
                           goes_data_path =goes_data_path, 
                           goes_row_pattern = '(M|X)[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', 
                           nb_frames_before_event = 48, 
                           sample_time = 1,
                           limit = None)
