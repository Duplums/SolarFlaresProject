import re, cv2, math
import numpy as np

class SF_picture:
    dict_segs = None
    header = None
    flare_events = None
    size = None
    attrString = '(Bp|Bp_err|Br|Br_err|Bt|Bt_err|continuum|[Dd]opplergram|magnetogram)'
    
    def __init__(self, dict_segs = {}, header = None, flare_events = None):
        assert type(dict_segs) is dict
        self.dict_segs = {}
        for k in dict_segs.keys():
            if(re.match(self.attrString, k)):
                data = np.array(dict_segs[k], dtype=np.float32)
                if(self.size is None):
                    self.size = list(data.shape)
                    self.dict_segs[k] = data
                elif(self.size != list(data.shape)):
                    print('Warning: segment {} does not have the same size as others. Ignored.'.format(k))
                else:
                    self.dict_segs[k] = data
            else:
                print('Warning: attribute {} not yet implemented'.format(k))
        self.header = header
        self.flare_events = flare_events
        
    def set_segments(self, key, data):
        if (re.match(self.attrString, key)):
            data = np.array(data, dtype=np.float32)
            if(self.size is None):
                self.size = list(data.shape)
                self.dict_segs[key] = data
            elif(self.size != list(data.shape)):
                print('Warning: segment {} does not have the same size as others. Ignored.'.format(key))
            else:
                self.dict_segs[key] = data
        else:
            print('Attribute {} not yet implemented :'.format(key))
    
    def get_segments(self, key):
        if (re.match(self.attrString, key)):
            return(self.dict_segs.get(key, None))
        else:
            print('Attribute {} not yet implemented :'.format(key))
            return None
 
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
        res = {'channels': self.dict_segs,'header':self.header, 'flare_events':self.flare_events}
        return res

# contains several frames of the same AR
class SF_video:
    frames = []
    frames_size = []
    flare_event = None
    
    def __init__(self, frames = [], frames_size = [], event = None):
        assert type(frames_size) is list and type(frames) is list
        assert np.all([type(frame) == SF_picture for frame in frames])
        assert len(frames) == len(frames_size)
        self.frames = frames
        self.frames_size = frames_size
        self.flare_event = event
    
    def length(self):
        return len(self.frames)
    
    def add_frame(self, SF_pic):
        if(type(SF_pic) is SF_picture):
            self.frames += [SF_pic]
            self.frames_size += [SF_pic.size]
        else:
            print('A frame must be a \'SF_picture\'. Ignored.')
            
    def display(self, height = 200, width = 400):
        nb_cols = 1
        if(len(self.frames) > 0):
            nb_cols = math.ceil(math.sqrt(len(self.frames[0].dict_segs)))
        count_seg = 0
        # Create the windows
        for frame in self.frames:
            for seg in frame.dict_segs.keys(): 
                count_seg += 1
                cv2.namedWindow(str(seg), cv2.WINDOW_NORMAL)
                cv2.resizeWindow(str(seg), width, height)
                cv2.moveWindow(str(seg), count_seg*width, (count_seg%nb_cols)*height)
            count_seg = 0
        # Display videos
        for frame in self.frames:
            for seg in frame.dict_segs.keys():
                cv2.imshow(str(seg), frame.dict_segs[seg])
            cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        
    def to_dict(self):
        res = {'flare_event':self.flare_event, 'frames': [pic.to_dict() for pic in self.frames], 'frames_size': self.frames_size}
        return res


#def picture_show(pics, segs = 'Bp,Bp_err,Br,Br_err,Bt,Bt_err,Dopplergram,magnetogram,continuum'):
#    segs = re.split(' ?, ?', segs)
#    flare_class = pics['flare_events'][0]['event_class']
#    peak = pics['flare_events'][0]['peak_time']
#    trec = pics['header']['T_REC']
#    noaa = pics['header']['NOAA_ARS']
#    harp = pics['header']['HARPNUM']
#    nb_channels = len(segs)
#    # Print every channels of an image
#    fig, axes = plt.subplots(int(nb_channels/3)+(nb_channels%3>0), \
#                             int(3*(nb_channels>=3)+nb_channels*(nb_channels<3)),\
#                             figsize=(15, 7.5))
#    for i, ax in enumerate(axes.flat):
#        if(i < nb_channels):
#        # Plot image.
#            ax.imshow(pics[segs[i]])
#            ax.set_xlabel('{}'.format(segs[i]))
#            # Remove ticks from the plot.
#            ax.set_xticks([])
#            ax.set_yticks([])
#    fig.suptitle('AR {} (HARP {}) taken on {}\n with peak flux magnitude {}\n(peak time: {})'.\
#                 format(noaa, harp, trec, flare_class, peak), fontsize=14, fontweight='bold')
#    fig.subplots_adjust(hspace=0.1, wspace=0.1)
#    plt.show()











    
