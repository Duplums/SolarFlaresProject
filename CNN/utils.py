
            # CONFIG FOR SOLAR FLARES DATA
config = { 'SF': {'data_dims': [None, None, 3], # MAX DIM on 7/20 : [2860, 2587] 
                  'batch_memsize': 2048 , # xMB / global batch
                  'num_threads' : 8,
                  'model': 'VGG_16_encoder_decoder',
                  'pb_kind': 'encoder',
                  'flare_level' : {'A': 1e0, 'B': 1e1, 'C': 1e2, 'M': 1e3, 'X': 1e4},
                  'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/SF',
                  'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_SF',
                  'input_features_dir' : '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset/SF',
                  'output_features_dir': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset/SF',
                  'regression_threshold': 6.9077552789, # == np.log(1e3) (== M-flare)
                  'learning_rate': 0.01,
                  'loss_weights': [1, 1.5],
                  'tolerance': 0.001,
                  'batch_norm': True,
                  'dropout_prob': 0.2,
                  'nb_classes': 2,
                  'batch_size': 1, # nb pictures / local batch,
                  'prefetch_buffer_size': 1, 
                  'num_epochs': 30, # nb total epochs
                  'checkpoint_iter': 1000, # save every 'checkpoint_iter' global iteration
                  'ar_attrs' : ['T_REC', 'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT',
                                'SIZE', 'SIZE_ACR', 'NACR', 'NPIX', 'LAT_MIN', 'LAT_MAX',
                                'LON_MIN', 'LON_MAX'],
                  'segs': ['Bp', 'Br', 'Bt'],
                  'goes_attrs' : ['event_class', 'noaa_active_region', 'event_date', 'start_time', 'end_time', 'peak_time'],
                  'subsampling' : 1,
                  'resize_method': 'NONE',
                  'rescaling_factor': 1,
                  'display' : False,
                  'time_step': 60, # time step used in each video
                  'training_paths': '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/train',
                  'testing_paths': '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/test'        
                  },
            # CONFIG FOR SOLAR FLARE FEATURES EXTRACTED FROM THE ENCODER
            'SF_encoded': {'data_dims': [None, None, None, 512],
                  'batch_memsize': 5000 , # xMB / global batch
                  'num_threads' : 48,
                  'model': 'LRCN',
                  'pb_kind': 'regression',
                  'flare_level' : {'A': 1e0, 'B': 1e1, 'C': 1e2, 'M': 1e3, 'X': 1e4},
                  'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/SF',
                  'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_SF',
                  'input_features_dir' : '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset/SF',
                  'output_features_dir': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset/SF',
                  'regression_threshold': 6.9077552789, # == np.log(1e3) (== M-flare)
                  'learning_rate': 0.01,
                  'loss_weights': [1, 1.5],
                  'tolerance': 0.001,
                  'batch_norm': True,
                  'dropout_prob': 0.2,
                  'nb_classes': 2,
                  'batch_size': 1, # nb videos / batch,
                  'prefetch_buffer_size': 1, 
                  'num_epochs': 50, # nb total epochs
                  'checkpoint_iter': 1000, # save every 'checkpoint_iter' global iteration
                  'ar_attrs' : [],
                  'segs': ['filter_{}'.format(k) for k in range(512)],
                  'subsampling' : 1,
                  'resize_method': 'ZERO_PADDING',
                  'rescaling_factor': 1,
                  'display' : False,
                  'time_step': 60, # time step used in each video (in minutes)
                  'training_paths': ['/home/data_encoded/train/B-class-flares',
                                     '/home/data_encoded/train/M-X-class-flares'],
                  'testing_paths': ['/home/data_encoded/test/B-class-flares',
                                    '/home/data_encoded/test/M-X-class-flares']                
                  },
           # CONFIG FOR MNIST DATA
          'MNIST': {'data_dims': [64, 64, 1],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'num_threads' : 8,
                      'model': 'VGG_16_encoder_decoder',
                      'pb_kind': 'encoder',
                      'checkpoint': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Checkpoints/MNIST',
                      'tensorboard': '/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Tensorboard/logs_MNIST',
                      'learning_rate': 0.01,
                      'loss_weights': 10*[1],
                      'tolerance': 0.01,
                      'batch_norm': True,
                      'dropout_prob': 0.3,
                      'nb_classes': 10,
                      'batch_size': 128, # nb pictures / batch,
                      'prefetch_buffer_size': 128,
                      'resize_method': 'LIN_RESIZING',
                      'segs': ['Nb'],
                      'num_epochs': 10,
                      'display' : True,
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'training_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/MNIST-HDF5/train'],
                      'testing_paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/MNIST-HDF5/test'],
                      'subsampling':1
                      },   
          # CONFIG FOR CIFAR10
          'CIFAR-10': {'data_dims': [32, 32, 3],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'num_threads' : 8,
                      'model': 'VGG_16',
                      'checkpoint': '/nobackup/bdufumie/SolarFlaresProject/Checkpoints/CIFAR10',
                      'tensorboard': '/nobackup/bdufumie/SolarFlaresProject/Tensorboard/CIFAR10',
                      'learning_rate': 0.01,
                      'loss_weights': 10*[1],
                      'tolerance': 0.001,
                      'batch_norm': True,
                      'dropout_prob': 0.3,
                      'nb_classes': 10,
                      'batch_size': 256, # nb pictures / batch,
                      'prefetch_buffer_size': 256,
                      'resize_method': 'NONE',
                      'segs': ['R', 'G', 'B'],
                      'num_epochs': 1, # nb batchs considered 
                      'display' : True,
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'training_paths': [''],
                      'testing_paths': [''],
                      'subsampling':1
                      }  
          }
