
            # CONFIG FOR SOLAR FLARES DATA
config = { 'SF': {'data_dims': [None, None, 3], # MAX DIM on 7/20 : [2860, 2587] 
                  'batch_memsize': 2048, # 4GB / global batch
                  'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/SF',
                  'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_SF',
                  'features_dir' : '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset/SF',
                  'learning_rate': 0.01,
                  'tolerance': 0.001,
                  'batch_norm': True,
                  'nb_classes': 2,
                  'batch_size': 2, # nb pictures / local batch
                  'num_epochs': 100, # nb global batchs considered 
                  'checkpoint_iter': 50, # save every 'checkpoint_iter' global iteration
                  'ar_attrs' : ['T_REC', 'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT',
                                'SIZE', 'SIZE_ACR', 'NACR', 'NPIX', 'LAT_MIN', 'LAT_MAX',
                                'LON_MIN', 'LON_MAX'],
                  'segs': ['Bp', 'Br', 'Bt'],
                  'goes_attrs' : ['event_class', 'noaa_active_region', 'event_date', 'start_time', 'end_time', 'peak_time'],
                  'subsampling' : 5,
                  'resize_method': 'NONE',
                  'time_step': 60, # time step used in each video
                  'training_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/train/B-class-flares', 
                                     '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/train/M-X-class-flares'],
                  'testing_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/test/B-class-flares', 
                                     '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/test/M-X-class-flares'],              
                  },
           # CONFIG FOR MNIST DATA
          'MNIST': {'data_dims': [28, 28, 1],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/MNIST',
                      'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_MNIST',
                      'learning_rate': 0.01,
                      'tolerance': 0.01,
                      'batch_norm': True,
                      'nb_classes': 10,
                      'batch_size': 256, # nb pictures / batch
                      'num_epochs': 1, # nb batchs considered 
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'segs': ['digit'],
                      'training_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/MNIST-HDF5/train'],
                      'testing_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/MNIST-HDF5/test'],
                      'subsampling':1
                      },   
          # CONFIG FOR CIFAR10
          'CIFAR-10': {'data_dims': [32, 32, 3],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/CIFAR10',
                      'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_CIFAR10',
                      'learning_rate': 0.01,
                      'tolerance': 0.001,
                      'batch_norm': True,
                      'nb_classes': 10,
                      'batch_size': 256, # nb pictures / batch
                      'num_epochs': 1, # nb batchs considered 
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'segs': ['R', 'G', 'B'],
                      'training_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/CIFAR10-HDF5/train'],
                      'testing_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/CIFAR10-HDF5/test'],
                      'subsampling':1
                      }  
          }
