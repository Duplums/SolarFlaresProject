from astropy.io import fits
from sunpy.time import TimeRange
import sunpy.instr.goes as goes_db
from datetime import timedelta
import drms, h5py, cv2, math
import os, csv, traceback, re, glob, sys
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
            self.goes_attrs = ['noaa_active_region']
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
    
        
    # For all files found, counts the number of videos, frames and channels/frames
    # as well as the mean size and diff between max and min size for each video.
    @staticmethod
    def check_statistics(path_to_files, out = None):
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
        results = {'nb_frames' : [], 'avg_size': [], 'nb_channels' : [], 'min_max_size': []}
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
                            for frame_key in db[vid_key].keys():
                                if('channels' in db[vid_key][frame_key].keys() and
                                    len(db[vid_key][frame_key]['channels'].shape) == 3):
                                    max_size = max(max_size, np.prod(db[vid_key][frame_key]['channels'].shape[0:2]))
                                    min_size = min(min_size, np.prod(db[vid_key][frame_key]['channels'].shape[0:2]))
                                    nb_channels = db[vid_key][frame_key]['channels'].shape[2]
                                    avg_size += db[vid_key][frame_key]['channels'].shape[0:2]
                                    nb_frames += 1
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
        out.write('NB OF VIDEOS : {}\n'.format(glob_counter))
        out.write('NB OF FRAMES : {}\n'.format(sum(results['nb_frames'])))
        out.write('NB OF FRAMES / VIDEO:\n')
        out.write('\t'+str(stats.describe(results['nb_frames'])))
        out.write('\nSIZE OF VIDEOS:\n')
        out.write('\t'+str(stats.describe(results['avg_size'])))
        out.write('\nMAX-MIN SIZE OF VIDEOS:\n')
        out.write('\t'+str(stats.describe(results['min_max_size'])))
        out.write('\nNB OF CHANNELS:\n')
        out.write('\t'+str(stats.describe(results['nb_channels'])))
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
                        out.write('\t\'{}\' => {}\n'.format(vid_key, db[vid_key].attrs['peak_time']))
            except:                
                print('Impossible to display peak time for file {}'.format(file))
                print(traceback.format_exc())
        if(close_flag):
            out.close()
           
    
    
    # Display a video from .hdf5 file
    @staticmethod
    def display_vid(file, vid):
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
                            cv2.namedWindow(channel.decode(), cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(channel.decode(), height, width)
                            cv2.imshow(channel.decode(), video[frame_key]['channels'][:,:,channel_count])
                            channel_count += 1
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
    
    # Get the data from the JSOC data base according to the solar eruptions described
    # in the GOES data base. The queries are based on SunPy and
    # the output files are in HDF5 format. The attributes associated to each 
    # pictures and downloaded are stored in self.ar_attrs. The number of channels
    # per picture is defined thanks to the attribute self.ar_segs.
    # Returns True if the data has been dowloaded successfully; False otherwise.
    def download_jsoc_data(self, files_core_name = 'jsoc_data',
                           directory = None,
                           goes_data_path = None, 
                           goes_row_pattern = '(B|C|M|X)[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', 
                           start_time = None, end_time = None,
                           hours_before_event = 24, sample_time = '@1h',
                           limit = 400):
        
        if(directory is None and not os.path.isdir(os.path.join(self.main_path, 'JSOC-Data'))):
            os.mkdir(os.path.join(self.main_path, 'JSOC-Data'))
            os.chdir(os.path.join(self.main_path, 'JSOC-Data'))
        elif(os.path.isdir(os.path.join(self.main_path, directory))):
            os.chdir(os.path.join(self.main_path, directory))
        else:
            print('The path {} does not exist.'.format(os.path.join(self.main_path, directory)))
            return False
        
        essential_ar_attrs = {'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT'}
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
        print('Look up of pictures until {}h before an event.'.format(hours_before_event))
        

        with open(goes_data_path, 'r', newline='') as file:        
            reader = csv.reader(file, delimiter=',')
            client = drms.Client()
            mem = 0 # Set a counter for the current cache memory (in bytes) used by videos
            part_counter = 54
            vid_counter = 0
            counter = 0
            current_save_file = h5py.File('{}_part_{}.hdf5'.format(files_core_name, part_counter), 'w') 
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
                        start_time = peak_time - timedelta(hours=hours_before_event)
                        # Change the date format
                        peak_time = self._UTC2JSOC_time(str(peak_time))
                        start_time = self._UTC2JSOC_time(str(start_time))
                        # Do the request to JSOC database
                        query = '{}[{}-{}{}]'.format(jsoc_serie, start_time, peak_time, sample_time)
                        if(len(self.ar_segs)==0): keys = client.query(query, key=self.ar_attrs)
                        else: keys, segments = client.query(query, key=self.ar_attrs, seg=self.ar_segs)
                        try:
                            # Downloads the video of this solar flare and construct 
                            # the HDF5 file.
                            nb_frame = len(keys.NOAA_AR)-1
                            dumping = False
                            current_vid = current_save_file.create_group('video{}'.format(vid_counter))
                            vid_counter += 1
                            for k in range(len(self.goes_attrs)):
                                current_vid.attrs[self.goes_attrs[k]] = event[k]
                            
                            print('Trying to extract data for video {} corresponding to event {}'.format(vid_counter, event[peak]))
                            frame_counter = 0
                            while(nb_frame > -1 and not dumping):
                                right_pic  = (keys.NOAA_AR[nb_frame] == ar_nb)\
                                   and abs(keys.LAT_FWT[nb_frame]) <= 68\
                                   and abs(keys.LON_FWT[nb_frame]) <= 68
                                
                                #Creates a new frame and add it to the video
                                if(right_pic):
                                    current_frame = current_vid.create_group('frame{}'.format(frame_counter))
                                    frame_counter += 1
                                    for k in range(len(self.ar_attrs)):
                                        current_frame.attrs[self.ar_attrs[k]] = keys[self.ar_attrs[k]][nb_frame]
                                    current_frame.attrs['SEGS'] = np.string_(list(self.ar_segs))
                                    data_shape = None # unknown 
                                    frame = None
                                    seg_counter = 0
                                    for seg in self.ar_segs:
                                        url = 'http://jsoc.stanford.edu' + segments[seg][nb_frame]
                                        data = np.array(fits.getdata(url), dtype=np.float32)
                                        if(data_shape is None):
                                            data_shape = data.shape
                                            frame = np.zeros(data_shape + (len(self.ar_segs),), dtype=np.float32)
                                        frame[:,:,seg_counter] = data
                                        seg_counter += 1
                                        mem += data.nbytes    
                                    current_frame.create_dataset('channels', data=frame)

                                    if(mem/(1024*1024) > 2*self.mem_limit):
                                        print('Memory usage > {}MB. Dumping...'.format(3*self.mem_limit))
                                        dumping = True
                                nb_frame -= 1
                            
                            print('Video {} associated to event {} extracted ({} frames)'.format(vid_counter, event[peak], frame_counter))

                        except: 
                            print('Impossible to extract data for event {0} (nb {1})'.format(event[peak], counter))
                            print(traceback.format_exc())
                else: # if the row pattern does not match 
                    print('Row ignored: '+str.join(',', event))
                
                if(int(counter*100.0/total_length)%5 == 0):
                    print(str(counter*100.0/total_length)+'% of GOES data set analyzed')
    
                # Save the current HDF5 file. Reset vid_counter for the next HDF5 file.
                if(mem/(1024*1024) > self.mem_limit):
                    current_save_file.close()
                    part_counter += 1
                    vid_counter = 0
                    mem = 0
                    current_save_file = h5py.File('{}_part_{}.hdf5'.format(files_core_name, part_counter), 'w') 
        
        # After the downloading, close the last file !
        current_save_file.close()
        print('Data base has been downloaded successfully !')
        return True


main_path = '/nobackup/bdufumie/SolarFlaresProject/Data/SF/'
goes_data_path = '/home6/bdufumie/SolarFlaresProject/DataQuery/GOES_dataset_B.csv'
goes_attrs = utils.config['SF']['goes_attrs']
ar_attrs = utils.config['SF']['ar_attrs']
ar_segs = utils.config['SF']['segs']

downloader = Data_Downloader(main_path, goes_attrs, ar_attrs, ar_segs)
downloader.download_jsoc_data(files_core_name = 'B_jsoc_data',
                           directory = 'B-class-flares-clean',
                           goes_data_path =goes_data_path, 
                           goes_row_pattern = 'B[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*', 
                           hours_before_event = 24, sample_time = '@1h',
                           limit = None)

