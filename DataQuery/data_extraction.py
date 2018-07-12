from astropy.io import fits
from sunpy.time import TimeRange
import sunpy.instr.goes as goes_db
from datetime import timedelta
import drms, h5py
import os, csv, sys, re
import numpy as np
import utils

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
            /frameN
                
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
            part_counter = 0
            vid_counter = 0
            counter = 0
            current_save_file = h5py.File(files_core_name+str(part_counter)+'.hdf5', 'w') 
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
                        
                        
                        # Downloads the video of this solar flare and construct 
                        # the HDF5 file.
                        try:
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
                                        current_frame.attrs[self.ar_attrs[k]] = keys[k]
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
                                        print('Memory usage > {}MB. Dumping...'.format(2*self.mem_limit))
                                        dumping = True
                                nb_frame -= 1
                            
                            print('Video {} associated to event {} extracted ({} frames)'.format(vid_counter, event[peak], frame_counter))

                        except: 
                            print('Impossible to extract data for event {0} (nb {1})'.format(start_time, counter))
                            print(sys.exc_info())
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
                    current_save_file = h5py.File(files_core_name+str(part_counter), 'w') 
        
        # After the downloading, close the last file !
        current_save_file.close()
        print('Data base has been downloaded successfully !')
        return True


#main_path = '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5'
#goes_data_path = '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/GOES_dataset.csv'
#goes_attrs = utils.config_SF['goes_attrs']
#ar_attrs = utils.config_SF['ar_attrs']
#ar_segs = utils.config_sf['ar_segs']
#
#downloader = Data_Downloader(main_path, goes_attrs, ar_attrs, ar_segs)
#downloader.download_jsoc_data(goes_data_path=goes_data_path, limit=None)




#### TEMP FUNCTIONS #####
def pyBin_to_hdf5(db, part):
    with h5py.File('jsoc_data_part_{}.hdf5'.format(part), 'w') as file:
        vid_counter  = 0
        for _, vid in db.items():
            video= file.create_group('video{}'.format(vid_counter))
            for attr, v in vid.flare_event.items():
                video.attrs[attr] = v
            for k in range(len(vid.frames)):
                f = vid.frames[k]
                frame = video.create_group('frame{}'.format(k))
                for key, v in f.header.items():
                    frame.attrs[key] = v
                frame.attrs['SEGS'] = np.string_(list(f.dict_segs.keys()))
                data = np.zeros(tuple(f.size)+(len(f.dict_segs),), dtype=np.float32)
                seg_count = 0
                for key, v in f.dict_segs.items():
                    data[:,:,seg_count] = v
                    seg_count += 1
                frame.create_dataset('channels', data=data, dtype=np.float32)
            vid_counter += 1























  
    
    
