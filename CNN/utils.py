
            # CONFIG FOR SOLAR FLARES DATA
config = { 'SF': {'data_dims': [256, 512, 9],
                  'batch_memsize': 4*1024, # 4GB / batch
                  'checkpoint': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Checkpoints_SF',
                  'tensorboard': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Tensorboard/logs_SF',
                  'learning_rate': 0.01,
                  'tolerance': 0.01,
                  'batch_norm': True,
                  'nb_classes': 2,
                  'batch_size': 10, # nb pictures / batch
                  'num_epochs': 20, # nb batchs considered 
                  'num_steps': 100, # nb step / batch (4GB)
                  'checkpoint_iter': 100, # save every 100 global iteration
                  'ar_attrs' : ['T_REC', 'NOAA_AR', 'HARPNUM', 'LAT_FWT', 'LON_FWT',
                                'SIZE', 'SIZE_ACR', 'NACR', 'NPIX', 'LAT_MIN', 'LAT_MAX',
                                'LON_MIN', 'LON_MAX'],
                  'segs': ['Bp', 'Bp_err', 'Br', 'Br_err', 'Bt', 'Bt_err', 'Dopplergram', 'continuum', 'magnetogram'],
                  'goes_attrs' : ['event_class', 'noaa_active_region', 'event_date', 'start_time', 'end_time', 'peak_time'],
                  'paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/positives', 
                            '/n/midland/w/dufumier/Documents/SolarFlaresProject/DataQuery/SF-HDF5/negatives']
                  },
           # CONFIG FOR MNIST DATA
          'MNIST': {'data_dims': [64, 64, 1],
                      'batch_memsize': 4*1024, # 4GB / batch
                      'checkpoint': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Checkpoints_MNIST',
                      'tensorboard': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Tensorboard/logs_MNIST',
                      'learning_rate': 0.01,
                      'tolerance': 0.01,
                      'batch_norm': True,
                      'nb_classes': 10,
                      'batch_size': 64, # nb pictures / batch
                      'num_epochs': 1, # nb batchs considered 
                      'num_steps': 500, # nb step / batch (4GB)
                      'checkpoint_iter': 500, # save every 500 global iteration
                      'segs': ['digit'],
                      'paths': ['/n/midland/w/dufumier/Documents/SolarFlaresProject/CNN/Dataset_MNIST']
                      }   
          }
