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


#shutil.rmtree(tensorboard_dir)
# It should contains several files 'features_patch[0..]' <-> 'labels_patch[0..]'
dataset_path = '/n/midland/w/dufumier/Documents/BlueSky/CNN/Dataset'
# Directory where we'll store the network internal variables
save_dir = '/n/midland/w/dufumier/Documents/BlueSky/CNN/Checkpoints'
# Name of the saving files
save_path = os.path.join(save_dir, 'basic_cnn')
# Directory where we'll store variable print in TensorBoard
tensorboard_dir = '/n/midland/w/dufumier/Documents/BlueSky/CNN/Tensorboard/logs/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

pictures_shape = (256, 512)
nb_channels = 9
test_size = 0.2
nb_classes = 2
dropout_rate = 0.2
plot_pics = False

(features, labels) = preproc.get_tf_dataset(path)
if(features is None):
    features, labels = preproc.create_tf_dataset(positive_path, negative_path, pictures_shape)
    preproc.save_tf_dataset(path, features, labels)
else:
    print('Dataset restored.')
    
if(plot_pics):
    preproc.plot_pictures([features[400][:,:,[0,2,4,6,7,8]]], [labels[500]])

#features = features[:,:,:,[0,2,4,6,7,8]]
s = rd.randint(0, 1000)
data_training_set, data_testing_set, label_training_set, label_testing_set = \
        train_test_split(features, labels, test_size=test_size, random_state=s)


print('Size of training set :\t{}'.format(len(data_training_set)))
print('Size of testing set:\t{}'.format(len(data_testing_set)))