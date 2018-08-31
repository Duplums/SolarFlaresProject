import tensorflow as tf
import time, os, traceback, argparse
import tracemalloc
from datetime import timedelta
import model, utils, data_gen
import numpy as np

# config: dict containing every info for training mode
# train_data_gen: load in RAM the data when needed

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
        print(traceback.format_exc())
        return False


def train_model(data):
    
    config = utils.config[data]
    checkpoint_dir = config['checkpoint']
    tensorboard_dir= config['tensorboard']
    learning_rate = config['learning_rate'] # initial learning rate
    epsilon = config['tolerance'] # useful for updating learning rate
    nb_classes = config['nb_classes']
    loss_weights = config['loss_weights']
    num_epochs = config['num_epochs']
    checkpoint_iter = config['checkpoint_iter']
    model_name = config['model']
    # Training graph
    G = tf.Graph()
    train_data_gen = data_gen.Data_Gen(data, config, G, max_pic_size=[3000,3000])
    
    with G.as_default():
        
        dyn_learning_rate = tf.placeholder(dtype=tf.float32,
                                           shape=[],
                                           name='learning_rate')
        
        input_data = tf.placeholder(dtype=tf.float32,
                                    shape=[None]+config['data_dims'],
                                    name='data')
        if(model_name == 'LSTM' or model_name == 'VGG_16'):
            input_labels = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='labels')
        if(model_name == 'LSTM'):
            input_seq_length = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name='seq_length')
        
        it_global = tf.Variable(tf.constant(0, shape=[], dtype=tf.int32), trainable=False)
        update_it_global = tf.assign_add(it_global, tf.shape(input_data)[0]) 
        if(model_name == 'VGG_16'):
            training_model = model.Model('VGG_16', nb_classes, batch_norm=config['batch_norm'], dropout_prob=config['dropout_prob'], loss_weights=loss_weights)
            training_model.build_vgg16_like(input_data)
            training_model.construct_results(input_labels)
        elif(model_name == 'LSTM'):
            training_model = model.Model('LSTM', nb_classes, loss_weights=loss_weights)
            training_model.build_lstm(input_data, input_seq_length)
            training_model.construct_results(input_labels)
        elif(model_name == 'VGG_16_encoder_decoder'):
            training_model = model.Model('VGG_16_encoder_decoder')
            training_model.build_vgg16_encoder_decoder(input_data)
            training_model.construct_results()
        elif(model_name == 'small_encoder_decoder'):
            training_model = model.Model('small_encoder_decoder')
            training_model.build_small_encoder_decoder(input_data)
            training_model.construct_results()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=dyn_learning_rate)
            grads = optimizer.compute_gradients(training_model.loss)
            grad_step = optimizer.apply_gradients(grads)
        
        merged = tf.summary.merge_all()
        global_init = tf.global_variables_initializer() # Every weights
        local_init = tf.local_variables_initializer() # For metrics
        saver = tf.train.Saver()

    # init and run training session
    print('Initializing training graph.')
    sess = tf.Session(graph=G, config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess=sess)
    sess.run(global_init)
    sess.run(local_init)
    start = time.time()
    tracemalloc.start()
    snap_old = tracemalloc.take_snapshot()
    with sess.as_default():
        restore_checkpoint(sess, saver, checkpoint_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir+'/train', sess.graph)        
        for epoch in range(num_epochs):
            # re-init the learning rate at the beginning of each epoch
            learning_rate = config['learning_rate']
            # re-init paths_to_file
            train_data_gen.init_paths_to_file()
            # training loop 
            features, labels, metadata = train_data_gen.gen_batch_dataset(save_extracted_data=False, 
                                                                             retrieve_data=False,
                                                                             take_random_files = True,
                                                                             get_metadata=True)
            batch_it = 0
            while(features is not None and len(features) > 0):
                # Create the TF input pipeline and preprocess the data
                train_data_gen.create_tf_dataset_and_preprocessing(features, labels)
                next_batch = train_data_gen.get_next_batch()
                # Computes every ops in each step   
                if(model_name in {'LSTM', 'VGG_16'}):
                    ops = [merged, grad_step, training_model.loss, training_model.prob, training_model.accuracy_up,
                           training_model.precision_up, training_model.recall_up, training_model.confusion_matrix_up, 
                           training_model.pred, update_it_global, it_global]
                elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
                    ops = [merged, grad_step, training_model.loss,
                           update_it_global, it_global]
                metrics = []
                if(model_name in {'LSTM', 'VGG_16'}):
                    metrics = [training_model.accuracy, 
                               training_model.precision, 
                               training_model.recall, 
                               training_model.confusion_matrix,
                               training_model.accuracy_per_class]
                step = 0
                end_of_data = False
                while not end_of_data:
                    try:
                        snap_new = tracemalloc.take_snapshot()
                        print(snap_new.compare_to(snap_old, 'lineno')[0])
                        snap_old = snap_new
                        data_train = sess.run(next_batch)
                        if(model_name == 'LSTM'):
                            inputs = {dyn_learning_rate : learning_rate,
                                      input_data : data_train[0][0],
                                      input_labels: data_train[0][1],
                                      input_seq_length : data_train[1]}
                        elif(model_name == 'VGG_16'):
                            inputs = {dyn_learning_rate : learning_rate,
                                      input_data : data_train[0],
                                      input_labels: data_train[1]}
                        elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
                            inputs = {dyn_learning_rate : learning_rate,
                                      input_data : data_train[0]}
                        #run_meta = tf.RunMetadata()
                        # runs the optimization and updates the metrics
                        results = sess.run(ops, feed_dict=inputs)
                        #                   options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        #                   run_metadata=run_meta)
                        #profiler.add_step(step, run_meta)
                        #option_builder = tf.profiler.ProfileOptionBuilder
                        
                        #opts = (option_builder(option_builder.time_and_memory()).
                        #    with_step(-1). # with -1, should compute the average of all registered steps.
                        #    with_file_output('test.txt').
                        #    select(['micros','bytes','occurrence']).order_by('bytes').
                        #    build())
                        # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
                        #profiler.profile_operations(options=opts)
                        # computes the metrics
                        metrics_ = sess.run(metrics)
                        # updates the learning rate and the old_loss
                        #if(results[-1] % 1000 == 0 and results[-1] > 0):
                        #    learning_rate /= 2
                        # plot the variables in TensorBoard
                        train_writer.add_summary(results[0], global_step=results[-1])
                        # plot in console the metrics we want and hyperparameters
                        if(model_name in {'LSTM', 'VGG_16'}):
                            print('Epoch {}, Batch {}, step {}, accuracy : {}, loss : {}, learning_rate : {}'.format(epoch, batch_it, step,
                                  metrics_[0], results[2], learning_rate))
                        elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
                            print('Epoch {}, Batch {}, step {}, loss : {}, learning_rate : {}'.format(epoch, batch_it, step,
                                  results[2], learning_rate))
                        # save the weigths
                        if((step*config['batch_size']) % 500  == 0 and step > 0):
                            saver.save(sess, os.path.join(checkpoint_dir,'training_{}.ckpt'.format(model_name)), it_global)
                            print('Checkpoint saved')
                        step += 1
                    except tf.errors.OutOfRangeError:                            
                        end_of_data = True
                # Load the next batch of data in memory
                features.clear()
                labels.clear()
                features, labels = train_data_gen.gen_batch_dataset(save_extracted_data=False, 
                                                                    retrieve_data=False,
                                                                    take_random_files=True,
                                                                    get_metadata=False)
                batch_it += 1
            saver.save(sess, os.path.join(checkpoint_dir,'training_{}.ckpt'.format(model_name)), it_global)
            
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return training_model

# Test the model created during the training phase. 
# If 'save_features' == True, save the features extracted
# by the CNN as a list of Tensor (np array) of dimension:
# nb_time_step x n_features where n_features = output space dim
            
def test_model(data, test_on_training = False, save_features = False):
    config = utils.config[data]
    checkpoint_dir = config['checkpoint']
    nb_classes = config['nb_classes']
    model_name = config['model']
    # Testing graph
    G = tf.Graph()
    test_data_gen = data_gen.Data_Gen(data, config, G, training=test_on_training,
                                      max_pic_size=[3000,3000])
    with G.as_default():        
        input_data = tf.placeholder(dtype=tf.float32,
                                    shape=[None]+config['data_dims'],
                                    name='data')
        input_labels = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name='labels')
        if(model_name == 'LSTM'):
            input_seq_length = tf.placeholder(dtype=tf.int32,
                                              shape=[None],
                                              name='seq_length')
        
        it_global = tf.Variable(tf.constant(0, shape=[], dtype=tf.int32), trainable=False)
        update_it_global = tf.assign_add(it_global, tf.shape(input_data)[0]) 
        if(model_name == 'VGG_16'):
            testing_model = model.Model('VGG_16', nb_classes, training_mode=False)
            testing_model.build_vgg16_like(input_data)
        elif(model_name == 'LSTM'):
            testing_model = model.Model('LSTM', nb_classes, training_mode=False)
            testing_model.build_lstm(input_data, input_seq_length)
        
        testing_model.construct_results(input_labels)
        saver = tf.train.Saver()
        local_init = [tf.local_variables_initializer(), it_global.initializer]
        
    # Init the graph with the last checkpoint.
    config = tf.ConfigProto()
    #config.inter_op_parallelism_threads = config['num_threads'] 
    sess = tf.Session(graph=G, config=config)
    tf.train.start_queue_runners(sess=sess)
    start = time.time()
    print('Checkpoint directory : {}'.format(checkpoint_dir))
    with sess.as_default():
        if(restore_checkpoint(sess, saver, checkpoint_dir)):
            # Do not forget to reset metrics before using it !
            sess.run(local_init)
            sess.run(testing_model.reset_metrics())
            # Load the first batch of data in memory
            features, labels, metadata = test_data_gen.gen_batch_dataset(save_extracted_data = False, 
                                                                         retrieve_data =  False,
                                                                         take_random_files = False,
                                                                         get_metadata=True)
            batch_it = 0
            while(features is not None and len(features) > 0):
                # Create the TF input pipeline and preprocess the data
                test_data_gen.create_tf_dataset_and_preprocessing(features, labels, metadata)
                next_batch = test_data_gen.get_next_batch()
                # Testing loop 
                end_of_data = False
                while(not end_of_data):
                    try:
                        data_test = sess.run(next_batch)
                        if(model_name == 'LSTM'):
                            inputs = {input_data : data_test[0][0],
                                      input_labels: data_test[0][1],
                                      input_seq_length : data_test[1]}
                        else:
                            inputs = {input_data : data_test[0],
                                      input_labels: data_test[1]}
                        ops = [testing_model.accuracy_up, testing_model.precision_up,
                               testing_model.recall_up, testing_model.confusion_matrix_up,
                               testing_model.prob, update_it_global, it_global]
                        
                        # we also want the output of our network
                        if(save_features):
                            ops += [testing_model.dense2]
                        
                        metrics = [testing_model.accuracy, testing_model.precision, 
                                   testing_model.recall, testing_model.accuracy_per_class,
                                   testing_model.confusion_matrix]
                        # Computes the output, updates the metrics and global iterator
                        res = sess.run(ops, feed_dict=inputs)
                        # add the features to a queue before its real saving on disk
                        if(save_features):
                            test_data_gen.add_output_features(res[-1], data_test[1], data_test[2])
                        # computes the metrics
                        metrics_ = sess.run(metrics)
                        #print('Global iteration {}'.format(res[6]))
                        #print('Proba : {}'.format(res[4]))
                        # updates the data in batch
                    except tf.errors.OutOfRangeError:
                        end_of_data = True
                
                # Dump the features in the queue (cannot be done before because
                # we do not control which pictures are given to the network)
                if(save_features):
                    test_data_gen.dump_output_features()
                # plots in console the metrics we want and hyperparameters
                print('-----\nBATCH {} -----\n'.format(batch_it))
                print('Accuracy : {}, Precision : {} \nRecall : {}, Accuracy per class : {}\nConfusion Matrix : {}\n-----'.format(metrics_[0], 
                  metrics_[1], metrics_[2], metrics_[3], metrics_[4]))
                if(test_on_training):
                    np.save(checkpoint_dir+'/training_confusion_matrix', metrics_[4])
                else:
                    np.save(checkpoint_dir+'/testing_confusion_matrix', metrics_[4])
                # Updates the data and the batch counter
                features, labels, metadata = test_data_gen.gen_batch_dataset(save_extracted_data = False, 
                                                                   retrieve_data = False,
                                                                   take_random_files = False, 
                                                                   get_metadata=True)
                batch_it += 1
        else:
            print('Impossible to test the model.')
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return testing_model

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type", type=str, help="Set the working data set.", choices=["SF", "SF_LSTM", "MNIST", "CIFAR-10"])
    parser.add_argument("--testing", help="Set the mode (training or testing mode).", default=False, action='store_true')
    parser.add_argument("--save_features", help="If this option is enabled, it saves the output features from the training or testing set.", default=False, action='store_true')
    parser.add_argument("--test_on_training", help="If this option and testing mode enabled, it tests the model on the training data set", default=False, action='store_true')
    parser.add_argument("--data_dims", nargs="+", help="Set the dimensions of feature ([H, W, C] for pictures) in the data set. None values accepted.")
    parser.add_argument("--batch_memsize", type=int, help="Set the memory size of each batch loaded in memory. (in MB)")
    parser.add_argument("-m", "--model", type=str, help="Set the neural network model used.", choices=["VGG_16", "LSTM", "VGG_16_encoder_decoder", "small_encoder_decoder"])
    parser.add_argument("-t", "--num_threads", type=int, help="Set the number of threads used for the preprocessing.")
    parser.add_argument("-c", "--checkpoint", type=str, help="Set the path to the checkpoint directory.")
    parser.add_argument("--tensorboard", type=str, help="Set the path to the tensorboard directory.")
    parser.add_argument("-r", "--resize_method", type=str, help="Set the resizing method.", choices=["NONE", "LIN_RESIZING", "QUAD_RESIZING", "ZERO_PADDING"])
    parser.add_argument("-b", "--batch_size", type=int, help="Set the number of features in each batch used during the training/testing phase.")
    parser.add_argument("-p", "--prefetch_batch_size", type=int, help="Set the number of pre-fetch features in each batch.")
    parser.add_argument("-s", "--subsampling", type=int, help="Set the subsampling value for each videos (only for SF data set).")
    parser.add_argument("-e", "--num_epochs", type=int, help="Set the total number of epochs for the training phase.")
    parser.add_argument("-w", "--loss_weights", nargs=2, type=float, help="Set the weights in the loss function (class imbalance problem).")
    parser.add_argument("--output_features_dir", type=str, help='Set the output directory where the extracted features should be saved.')
    args = parser.parse_args()
    data = args.data_type
    for key, val in args._get_kwargs():
        if(val is not None):
            if(key=='data_dims'): #Special case to accept 'None' values
                data_dims = []
                for k in val:
                    if(k == 'None'):
                        data_dims += [None]
                    else:
                        data_dims += [int(k)]
                utils.config[data][key] = data_dims
            elif(key!='data_type' and key!='testing'):
                utils.config[data][key] = val
    tf.reset_default_graph()
    if(args.testing):
        test_model(data, args.test_on_training, args.save_features)
    else:
        t = train_model(data)

