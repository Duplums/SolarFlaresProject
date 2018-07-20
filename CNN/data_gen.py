
'''
This class aims to create a list of files where we can find the data and to manage
the computer memory. It gives to a preprocessor the data when needed and generates
a batch of data. The data memory size (in MB) is given and should not be exceeded.

'''

import os
import numpy as np
import preprocessing as pp

class Data_Iterator:
    features, labels = None, None # Refs to the data 
    batch_size = None
    _it = None
    
    def __init__(self, x, y, batch_size):
        self.features= x
        self.labels = y
        self.batch_size = batch_size
        self._it = 0
    
    def is_next_batch(self):
        if(self.features is None):
            return True
        return len(self.features) > self._it 
    
    def next_batch(self):
        if(self.features is None or self.labels is None):
            return None
        res = (self.features[self._it:self._it+self.batch_size], self.labels[self._it:self._it+self.batch_size])
        self._it += self.batch_size
        return res
    
    def where_is_it(self):
        if(self.features is None):
            return 0
        if(self._it >= len(self.features)):
            return 1
        return float(self._it)/len(self.features)
    
    
class Data_Gen:
    
    paths_to_file = None # array of paths where we can find the data
    size_of_files = None
    memory_size = None
    data_dims = None 
    segs = None
    database = None
    subsampling = None
    preprocessor = None
    training_mode = None
    
    def __init__(self, data, config, training=True):
        self.paths_to_file = []
        self.size_of_files = []
        self.memory_size =  config['batch_memsize']
        self.data_dims = config['data_dims']
        self.training_mode = training
        if(training):
            self.subsampling = config['subsampling']
            paths = config['training_paths']
        else:
            self.subsampling = 1
            paths = config['testing_paths']

        if(data == 'SF'):
            self.preprocessor = pp.Preprocessor(data, [], self.subsampling, 
                                                config['nb_classes'], config['data_dims'][:2], 
                                                config['resize_method'], config['segs'],
                                                config['time_step'],
                                                config['features_dir'])
        else:
            self.preprocessor = pp.Preprocessor(data)

        # First check
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
    
    def next_batch(self, 
                   save_features = False, 
                   retrieve_features = False,
                   random = False):
        
        batch_mem = 0
        if(random):
            files_index = np.random.permutation(len(self.size_of_files))
        else:
            # Heuristic to know which files we will consider
            files_index = np.argsort(self.size_of_files)
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
        self.preprocessor.set_files(files_in_batch)
        print('Files given to the preprocessor : {}'.format(files_in_batch))
        return self.preprocessor.extract_features(save_features, retrieve_features)        
    
    @staticmethod
    def get_random_batch(x, y, batch_size):
        idx = np.random.choice(len(x),
                               size=batch_size,
                               replace=False)
        x_batch = x[idx]
        y_batch = y[idx]
        return (x_batch, y_batch)
                
