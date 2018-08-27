
            # CONFIG FOR SOLAR FLARES DATA
config = { 'SF': {'data_dims': [256, 512, 3], # MAX DIM on 7/20 : [2860, 2587] 
                  'batch_memsize': 2048, # xMB / global batch
                  'num_threads' : 8,
                  'model': 'VGG_16',
                  'checkpoint': '/nobackup/bdufumie/SolarFlaresProject/Checkpoints/SF',
                  'tensorboard': '/nobackup/bdufumie/SolarFlaresProject/Tensorboard/SF',
                  'input_features_dir' : '/nobackup/bdufumie/SolarFlaresProject/Dataset/SF',
                  'output_features_dir': '/nobackup/bdufumie/SolarFlaresProject/Data/SF_LSTM/train',
                  'learning_rate': 0.01,
                  'tolerance': 0.001,
                  'batch_norm': True,
                  'dropout_prob': 0.3,
                  'nb_classes': 2,
                  'batch_size': 2, # nb pictures / local batch,
                  'prefetch_buffer_size': 2, 
                  'num_epochs': 10, # nb global batchs considered 
                  'checkpoint_iter': 100, # save every 'checkpoint_iter' global iteration
                  'ar_attrs' : ['T_REC', 'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT',
                                'SIZE', 'SIZE_ACR', 'NACR', 'NPIX', 'LAT_MIN', 'LAT_MAX',
                                'LON_MIN', 'LON_MAX'],
                  'segs': ['Bp', 'Br', 'Bt'],
                  'goes_attrs' : ['event_class', 'noaa_active_region', 'event_date', 'start_time', 'end_time', 'peak_time'],
                  'subsampling' : 1,
                  'resize_method': 'LIN_RESIZING',
                  'time_step': 60, # time step used in each video
                  'training_paths': ['/nobackup/bdufumie/SolarFlaresProject/Data/SF/train/B-class-flares', 
                                     '/nobackup/bdufumie/SolarFlaresProject/Data/SF/train/M-X-class-flares'],
                  'testing_paths': ['/nobackup/bdufumie/SolarFlaresProject/Data/SF/test/B-class-flares', 
                                     '/nobackup/bdufumie/SolarFlaresProject/Data/SF/test/M-X-class-flares'],              
                  },
            # CONFIG FOR SOLAR FLARE FEATURES EXTRACTED FROM THE CNN
            'SF_LSTM': {'data_dims': [None,  1024], # MAX_T_STEP x N_INPUT
                  'batch_memsize': 1024, # xMB / global batch
                  'num_threads' : 8,
                  'model': 'LSTM',
                  'checkpoint': '/nobackup/bdufumie/SolarFlaresProject/Checkpoints/SF_LSTM',
                  'tensorboard': '/nobackup/bdufumie/SolarFlaresProject/Tensorboard/SF_LSTM',
                  'learning_rate': 0.01,
                  'tolerance': 0.001,
                  'nb_classes': 2,
                  'batch_size': 256, # nb sequences / local batch
                  'prefetch_buffer_size': 2, 
                  'num_epochs': 15, # nb global batchs considered 
                  'checkpoint_iter': 50, # save every 'checkpoint_iter' global iteration                
                  'time_step': 60, # time step used in each video
                  'training_paths': ['/nobackup/bdufumie/SolarFlaresProject/Data/SF_LSTM/train'],
                  'testing_paths': ['/nobackup/bdufumie/SolarFlaresProject/Data/SF_LSTM/test'],              
                  },
           # CONFIG FOR MNIST DATA
          'MNIST': {'data_dims': [28, 28, 1],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'num_threads' : 8,
                      'model': 'VGG_16',
                      'checkpoint': '/nobackup/bdufumie/SolarFlaresProject/Checkpoints/MNIST',
                      'tensorboard': '/nobackup/bdufumie/SolarFlaresProject/Tensorboard/MNIST',
                      'learning_rate': 0.01,
                      'tolerance': 0.01,
                      'batch_norm': True,
                      'dropout_prob': 0.3,
                      'nb_classes': 10,
                      'batch_size': 256, # nb pictures / batch,
                      'prefetch_buffer_size': 256, 
                      'num_epochs': 1, # nb batchs considered 
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'training_paths': [''],
                      'testing_paths': [''],
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
                      'tolerance': 0.001,
                      'batch_norm': True,
                      'dropout_prob': 0.3,
                      'nb_classes': 10,
                      'batch_size': 256, # nb pictures / batch,
                      'prefetch_buffer_size': 256,
                      'num_epochs': 1, # nb batchs considered 
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'training_paths': [''],
                      'testing_paths': [''],
                      'subsampling':1
                      }  
          }
