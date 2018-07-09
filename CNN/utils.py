import tensorflow as tf

def restore_checkpoint(session, saver, save_dir):
    try:
        print("Trying to restore last checkpoint ...")
    
        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    
        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=last_chk_path)
    
        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
        return True
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint.")
        return False


config = {'data_dims': [256, 512, 9],
          'batch_memsize': 4*1024, # 4GB / batch
          'checkpoint': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Checkpoints',
          'tensorboard': '/n/midland/w/dufumier/Documents/BlueSky/CNN/Tensorboard/logs',
          'learning_rate': 0.01,
          'batch_norm': True,
          'nb_classes': 2,
          'batch_size': 10, # nb pictures / batch
          'num_epochs': 20, # nb batchs considered 
          'num_steps': 100, # nb step / batch (4GB)
          'checkpoint_iter': 100, # save every 100 global iteration
          'segments': ['Bp', 'Bp_err', 'Br', 'Br_err', 'Bt', 'Bt_err', 'Dopplergram', 'continuum', 'magnetogram']
          }    

paths = ['/n/midland/w/dufumier/Documents/BlueSky/DataQuery/SF-negative', '/n/midland/w/dufumier/Documents/BlueSky/DataQuery/SF-positive']
