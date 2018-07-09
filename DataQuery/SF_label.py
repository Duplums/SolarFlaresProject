from astropy.io import fits
from sunpy.time import TimeRange
import sunpy.instr.goes as goes_db
from sunpy.net import jsoc
from sunpy.net import attrs as attrs
from sunpy.time import parse_time
from datetime import timedelta
import skimage.transform as sk
import drms
import os, glob, re, csv
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class SF_picture:
    Bp = None
    Bp_err = None
    Br = None
    Br_err = None
    Bt = None
    Bt_err = None
    continuum = None
    dopplergram = None
    magnetogram = None
    header = None
    flare_events = None
    attrString = '(Bp|Bp_err|Br|Br_err|Bt|Bt_err|continuum|[Dd]opplergram|magnetogram)'
    def __init__(self, Bp = None, Bp_err = None, Br = None, Br_err = None, \
                 Bt = None, Bt_err = None, continuum = None, dopplergram = None, \
                 magnetogram = None, header = None, flare_events = None):
        self.Bp= Bp
        self.Br = Br
        self.Bt = Bt
        self.Bp_err = Bp_err
        self.Br_err = Br_err
        self.Bt_err = Bt_err
        self.continumm = continuum
        self.dopplergram = dopplergram
        self.magnetogram = magnetogram
        self.header = header
        self.flare_events = flare_events
        
    def setAttr(self, txt, data):
        if (not re.match(self.attrString, txt)):
            print('Attribute not yet implemented :'+txt)
        else:
            if txt == 'Bp': 
                self.Bp = data
            elif txt == 'Bp_err':
                self.Bp_err= data
            elif txt == 'Bt':
                self.Bt = data
            elif txt == 'Bt_err':
                self.Bt_err = data
            elif txt == 'Br': 
                self.Br = data
            elif txt == 'Br_err':
                self.Br_err= data
            elif txt == 'continuum':
                self.continuum = data
            elif re.match('[Dd]opplergram', txt):
                self.dopplergram = data
            elif txt == 'magnetogram':
                self.magnetogram = data
        
    def get_noaas_nb(self):
        if (self.header is None) or (self.header['NOAA_NUM'] is None): 
            return 0
        return self.header['NOAA_NUM']
    
    def get_noaas_ar_num(self):
        if (self.header is None) or (self.header['NOAA_ARS'] is None):
            return []
        return list(map(int, re.split(',', self.header['NOAA_ARS'])))

    def get_t_rec(self):
        if (self.header is None) or (self.header['T_REC'] is None):
            return ''
        else:
            return self.header['T_REC']
    
    # convert SF_picture to dictionnary
    def to_dict(self):
        res = {'Bp':self.Bp, 'Bp_err':self.Bp_err, 'Bt':self.Bt, 'Bt_err':self.Bt_err, \
            'Br':self.Br, 'Br_err':self.Br_err, 'continuum':self.continuum, \
            'Dopplergram':self.dopplergram, 'magnetogram':self.magnetogram,\
            'header':self.header, 'flare_events':self.flare_events}
        return res

# contains several frames of the same AR
class SF_video:
    frames = []
    flare_event = None
    
    def __init__(self, frames = [], event = None):
        self.frames = frames
        self.flare_event = event
    
    def length(self):
        return len(self.frames)
    
    def add(self, SF_pic):
        self.frames += [SF_pic]
        
    def to_dict(self):
        res = {'flare_event':self.flare_event, 'frames' : [pic.to_dict() for pic in self.frames]}
        return res
    
    
# get every GOES events from tstart to tend and store it into .csv file.
# Format: class, NOAA, event_date, start_time, end_time, peak_time
# Eventually, the user can provide GOES data
def import_GOES_dataset(csv_file_name, path, goes_events = None, tstart = None, tend = None):
    os.chdir(path)
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['class', 'NOAA_ar_num', 'event_date', 'start_time', 'end_time', 'peak_time'])
        if(goes_events is None):
            goes_events = goes_db.get_goes_event_list(TimeRange(tstart, tend))
        for event in goes_events:
            if(event['noaa_active_region'] > 0):
                writer.writerow([event['goes_class'], event['noaa_active_region'], \
                             event['event_date'], event['start_time'], \
                             event['end_time'], event['peak_time']])


# Get the solar flare (SF) pictures from SDO/HMI JSOC database labelled with 
# the GOES database. The label is based on the GOES X-ray flux peak. 
# Example: positive_class = ['M', 'X'], negative_class = ['B', 'C']
# nearest_SF_event: do we take only the nearest picture from a SF event ?
# if not, take all pictures within 24h before the event
def import_SF_dataset(goes_dataset_path, \
                      SF_dataset_path, \
                      flare_class = '(B|C|M|X)',\
                      segs_str = 'Bp,Bp_err,Br,Br_err,Bt,Bt_err,Dopplergram,magnetogram,continuum',\
                      keys_str = 'T_REC, NOAA_AR, HARPNUM, LAT_FWT, LON_FWT',\
                      nearest_SF_event = True,\
                      sample_time = '@1h',\
                      nb_parts = 10.0,\
                      limit = 400):
    
    assert keys_str.find('NOAA_AR') > -1 and keys_str.find('HARPNUM') > -1
    assert keys_str.find('LAT_FWT') > -1 and keys_str.find('LON_FWT') > -1
    
    database_serie = 'hmi.sharp_cea_720s[1-7256]' # every SF detected in SHARP.
    CLASS_INDEX , AR_INDEX, EVENT_DATE, START, END, PEAK = range(6)
    goes_row_pattern = flare_class+'[1-9]\.[0-9],[1-9][0-9]*,.*,.*,.*,.*'
    segs_str_array = re.split(' ?, ?', segs_str)
    keys_str_array = re.split(' ?, ?', keys_str)
    total_length = sum(1 for line in open(goes_dataset_path, 'r'))
    # Estimations...
    nb_positive = 0
    considered_events = []
    with open(goes_dataset_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        counter = 0
        for event in reader:
            counter += 1
            if(re.match(goes_row_pattern, str.join(',', event))):
                nb_positive += 1
                considered_events += [counter]
    
    if(limit is not None and nb_positive > limit):
        events_really_considered = np.random.choice(considered_events, limit)
    print(nb_positive)
    
    print('Nb of flares: '+str(nb_positive))
    print('Total size estimated: '+str(30*nb_positive)+'Mo')
    print('Size/part estimated: '+str(30*nb_positive/nb_parts)+'Mo')
    with open(goes_dataset_path, 'r', newline='') as file:        
        reader = csv.reader(file, delimiter=',')
        client = drms.Client()
        SF_dico = {}
        part_counter = 0
        counter = 0
        counter_videos = 0
        for event in reader:
            counter += 1
            if(re.match(goes_row_pattern, str.join(',', event)) and 
               (limit is None or nb_positive <= limit or
               (counter in events_really_considered))):
                ar_nb = int(event[AR_INDEX])
                counter_videos += 1
                # We process only numbered flares
                if(ar_nb > 0):
                    peak_time = drms.to_datetime(event[PEAK])
                    start_time = peak_time - timedelta(days=1)
                    # Change the date format
                    peak_time = UTC2JSOC_time(str(peak_time))
                    start_time = UTC2JSOC_time(str(start_time))
                    # Do the request to JSOC database
                    if(segs_str == ''):
                        keys = client.query(database_serie+'['+start_time+'-'+\
                                                 peak_time+sample_time+']',\
                                                 key = keys_str)
                    else:
                        keys, segments = client.query(database_serie+'['+start_time+'-'+\
                                                 peak_time+sample_time+']',\
                                                 key = keys_str, \
                                                 seg = segs_str)
                    # Downloads all pictures and store them together
                    try:
                        i = len(keys.NOAA_AR)-1
                        pic_not_found = True
                        dico_key = start_time + '_' + str(ar_nb)
                        SF_vid = SF_video([], {'event_class': event[CLASS_INDEX],\
                                               'event_date': event[EVENT_DATE], \
                                               'peak_time':event[PEAK],\
                                               'start_time':event[START],\
                                               'end_time':event[END]})
                        print("Flare event created for video {}".format(counter_videos))
                        while(i > -1 and (pic_not_found or not nearest_SF_event)):
                            pic_not_found = (keys.NOAA_AR[i] != ar_nb) \
                               or abs(keys.LAT_FWT[i]) > 68\
                               or abs(keys.LON_FWT[i]) > 68
                            if(not pic_not_found):
                                SF_pic = SF_picture()
                                # we download what we need
                                for seg in segs_str_array:
                                    if seg != '' :
                                        url = 'http://jsoc.stanford.edu' + segments[seg][i]
                                        data = np.array(fits.getdata(url), dtype=np.float32)
                                        SF_pic.setAttr(seg, data)
                                header = {}
                                for key in keys_str_array:
                                    header[key] = keys[key][i]
                                SF_pic.header = header                         
                                SF_vid.add(SF_pic)
                            i -= 1
                            SF_dico[dico_key] = SF_vid
                    except: 
                        print('Impossible to extract data for event {0} (nb {1})'.format(start_time, counter))
            else: # if the row pattern does not match 
                print('Row ignored: '+str.join(',', event))
            
            if(int(counter*100.0/total_length)%5 == 0):
                print(str(counter*100.0/total_length)+'% of GOES data set analyzed')
            print('Videos {} downloaded.'.format(counter_videos))
            print("Dico has length {}".format(len(SF_dico)))
            # Save a part of the database
            if(counter_videos % 10 == 0 and len(SF_dico) > 0): 
                db = {k : vid.to_dict() for k, vid in SF_dico.items()}
                with open(SF_dataset_path+'_part_'+str(part_counter), 'wb') as f:
                    pickle.dump(db,f)
                    SF_dico.clear()
                    db.clear()
                    part_counter += 1
                print("Memory dump - part {}".format(part_counter))
              
    db = {k : vid.to_dict() for k, vid in SF_dico.items()}
    with open(SF_dataset_path+'_part_'+str(part_counter), 'wb') as f:
        pickle.dump(db,f)
        SF_dico.clear()    

# create an other database with just the extracted features (no pictures)
#def collect_features(SF_dataset_path,\
#                     regex_name,\
#                     features_to_extract):
#    os.chdir(SF_dataset_path)
#    files = glob.glob(regex_name)
#    for file in files:
#         with open(file, 'rb') as f:
#            db = pickle.load(f)


#  Erase missing data ('NaN')
# BE CAREFUL AND DO A BACKUP OF THE DATASET BEFORE RUNNING IT
def clean_SF_dataset(SF_dataset_path, \
                     regex_name, \
                     segs_str = 'Bp,Bp_err,Br,Br_err,Bt,Bt_err,Dopplergram,magnetogram,continuum'):
    os.chdir(SF_dataset_path)
    files = glob.glob(regex_name)
    segs_str_array = re.split(',', segs_str)
    mem_save = 0 # Total memory saved (MB)   
    total_nb_pictures = 0
    total_nb_SF = 0
    for file in files:
        db = get_db(file)
        keys = db.keys()
        for key in keys:
            pic = db[key]
            total_nb_SF += 1
            for seg in segs_str_array:
                total_nb_pictures += 1
                # Each seg can be viewed as a picture 
                pic[seg] = np.array(pic[seg], dtype = np.float32)
                data = pic[seg]
                
                # STEP 1: erase 'NaN'
                where_is_nan = np.argwhere(np.isnan(data))
                if(len(where_is_nan) > 0):
                    nan_up_left_corner = (min(where_is_nan[:,0]), min(where_is_nan[:,1]))
                    nan_down_right_corner = (max(where_is_nan[:,0]), max(where_is_nan[:,1]))
                    # Select only the biggest rectangle that 
                    # not contains 'NaN'
                    data_shape = np.shape(data)
                    right_split_pic_width = nan_up_left_corner[1]
                    left_split_pic_width = data_shape[1] - nan_down_right_corner[1]
                    if(right_split_pic_width > left_split_pic_width):
                        # conserve only the right picture's part
                        pic[seg] = data[:,0:nan_up_left_corner[1]]
                    else:
                        #otherwise, conserve the other part
                        pic[seg] = data[:,nan_down_right_corner[1]+1:]
                    data_reshape = np.shape(pic[seg])
                    print('Picture '+key+', seg '+seg+' treated: '+\
                          str(data_shape)+' --> '+str(data_reshape))
                    mem_save += ((data_shape[0]-data_reshape[0])*data_shape[1] +\
                                (data_shape[1]-data_reshape[1])*data_reshape[0])*data.itemsize
                if(np.any(np.isnan(pic[seg]))):
                    print('Error for the following picture (file:'+file+')')
                    print('Before -->')
                    plt.imshow(data)
                    plt.show()
                    print('After cleaning -->')
                    plt.imshow(pic[seg])
                    plt.show()
                
        with open(file, 'wb') as f:
            pickle.dump(db, f)
    mem_save /= (1024*1024) # from Bytes to  MB
    print('Total number of pictures analyzed: '+str(total_nb_pictures))
    print('Total number of SF analyzed: '+str(total_nb_SF))
    print('Memory saved : '+str(mem_save)+'MB')

def UTC2JSOC_time(UTC):
    JSOC = re.sub('-', '.', UTC)
    JSOC = re.sub(' ' , '_', UTC) + '_TAI'
    return JSOC


# returns a list of SF_pictures from a list of fits files
def extract_HMI_SHARP_dataset(path, ignore_incorrect_files = True):
    # First, extract all the data from fits files
    os.chdir(path) # we set the right current directory
    # Now, iterate over all the files
    pics_dict = {} # Name <-> SF_picture
    possible_segments = SF_picture.attrString
    files = glob.glob('*.fits')
    count_progress = 0
    count_ignored_file = 0
    for file in files:
        # small tests
        if(not re.match('hmi\.sharp_cea_720s\.[1-9][0-9]*\..*\.'+\
                         possible_segments+'\.fits', file)):
            if (not ignore_incorrect_files):
                print('Error while scanning file '+file)
                return None
            else:
                count_ignored_file += 1
        else:
            hdu = fits.open(file)# Read the fits file
            if(len(hdu) != 2):
                print('Wrong size for fits file '+file)
                return None
            hdu[1].verify('silentfix') # fix header issues
            file_decomposed = re.split('\.', file)
            pic_name = file_decomposed[2] + file_decomposed[3]
            if(pic_name in pics_dict):
                SF_pic = pics_dict[pic_name]
                SF_pic.setAttr(file_decomposed[4], hdu[1].data)
            else:
                SF_pic = SF_picture()
                SF_pic.header = hdu[1].header
                SF_pic.setAttr(file_decomposed[4], hdu[1].data)
                pics_dict[pic_name] = SF_pic
            hdu.close()
        count_progress += 1
        if(int(count_progress*100.0/len(files)) % 10 == 0): #update every 10%
            print('Scanning '+str(int(count_progress*100.0/len(files)))+'%')
    print('Ignored files: '+str(count_ignored_file*100.0/len(files))+'%')
    # Every picture is ready.
    return (list(pics_dict.values()))


def labelled_SF_pictures(pictures):
    # we labelled each picture (AR) by using GOES database. We query this db when
    # a NOAA nb is associated to the AR we're looking. The label is defined with
    # the nearest GOES event within 24h after the picture has been taken. 
    
    match_count = 0
    for pic in pictures:
        # we look only pictures that have a NOAA nb
        pic.flare_events = []
        if(pic.get_noaas_nb() == 1):
            noaa_num = pic.header['NOAA_AR']
            tstart = parse_time(pic.get_t_rec())
            timeRange = TimeRange(tstart, timedelta(1))
            goes_events = goes_db.get_goes_event_list(timeRange)
            # Now look into goes event
            for event in goes_events:
                if event['noaa_active_region'] == noaa_num: # we find a match !
                    pic.flare_events += [{'event_class': event['goes_class'],
                                        'event_date': event['event_date'],
                                       'start_time': event['start_time'],
                                       'peak_time': event['peak_time'],
                                       'end_time': event['end_time']}]
                    match_count += 1
    print('Proportion of match found: '+str(match_count*100.0/len(pictures))+'%')

# Tranforms a binary file into a Python object
def get_db(file_path):
    with open(file_path, 'rb') as f:
        db = pickle.load(f)
    return db

def picture_show(pics, segs = 'Bp,Bp_err,Br,Br_err,Bt,Bt_err,Dopplergram,magnetogram,continuum'):
    segs = re.split(' ?, ?', segs)
    flare_class = pics['flare_events'][0]['event_class']
    peak = pics['flare_events'][0]['peak_time']
    trec = pics['header']['T_REC']
    noaa = pics['header']['NOAA_ARS']
    harp = pics['header']['HARPNUM']
    nb_channels = len(segs)
    # Print every channels of an image
    fig, axes = plt.subplots(int(nb_channels/3)+(nb_channels%3>0), \
                             int(3*(nb_channels>=3)+nb_channels*(nb_channels<3)),\
                             figsize=(15, 7.5))
    for i, ax in enumerate(axes.flat):
        if(i < nb_channels):
        # Plot image.
            ax.imshow(pics[segs[i]])
            ax.set_xlabel('{}'.format(segs[i]))
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('AR {} (HARP {}) taken on {}\n with peak flux magnitude {}\n(peak time: {})'.\
                 format(noaa, harp, trec, flare_class, peak), fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()
    
path = '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery'
tstart = '2010/05/1 00:00:00';
tend = '2018/06/20 00:00:00';
goes_dataset_path = path + '/GOES_dataset.csv'
SF_dataset_positive_path = path + '/SF-positive/SF_positive'
SF_dataset_negative_path = path + '/SF-negative/SF_negative'
regex_SF_name = 'SF_*_part_[0-9]*.*'

#import_GOES_dataset('GOES_dataset.csv', path, None, tstart, tend)
#import_SF_dataset(goes_dataset_path, SF_dataset_positive_path, '(M|X)')
#clean_SF_dataset(path+'/SF-positive-data-parts', regex_SF_name)
#import_SF_dataset(goes_dataset_path, SF_dataset_negative_path, 'B')
#clean_SF_dataset(path+'/SF-negative-data-parts', regex_SF_name)
#import_SF_dataset(goes_dataset_path, SF_dataset_positive_path, '(M|X)',\
#                  keys_str = 'T_REC, NOAA_AR, HARPNUM,LAT_FWT,LON_FWT,SIZE,SIZE_ACR,NACR,NPIX,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX',\
#                  nearest_SF_event = False,\
#                  sample_time='@1h',\
#                  nb_parts = 100,
#                  limit = None)

import_SF_dataset(goes_dataset_path, SF_dataset_positive_path, 'B',\
                  keys_str = 'T_REC, NOAA_AR, HARPNUM,LAT_FWT,LON_FWT,SIZE,SIZE_ACR,NACR,NPIX,LAT_MIN,LAT_MAX,LON_MIN,LON_MAX',\
                  nearest_SF_event = False,\
                  sample_time='@1h',\
                  nb_parts =1000,
                  limit = 2500) 
    
    
    
    
    
    
    
    
    
    
