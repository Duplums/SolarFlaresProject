import tensorflow as tf
import time, os, traceback
from datetime import timedelta
import model, utils, data_gen

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
            training_model = model.Model('VGG_16', nb_classes, batch_norm=config['batch_norm'], dropout_prob=config['dropout_prob'])
            training_model.build_vgg16_like(input_data)
        elif(model_name == 'LSTM'):
            training_model = model.Model('LSTM', nb_classes)
            training_model.build_lstm(input_data, input_seq_length)
        
        training_model.construct_results(input_labels)
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
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    sess.run(global_init)
    sess.run(local_init)
    start = time.time()
    with sess.as_default():
        restore_checkpoint(sess, saver, checkpoint_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir+'/train', sess.graph)
        profiler = tf.profiler.Profiler(sess.graph)
        # training loop 
        for epochs in range(num_epochs):
            # re-init the learning rate at the beginning of each epoch
            learning_rate = config['learning_rate']
            # Load the next batch of data in memory
            features, labels = train_data_gen.gen_batch_dataset(save_extracted_data=(model_name=='SF'), 
                                                                retrieve_data=(model_name=='SF'),
                                                                take_random_files=True,
                                                                get_metadata=False)
            # Create the TF input pipeline and preprocess the data
            train_data_gen.create_tf_dataset_and_preprocessing(features, labels)
            # Computes every ops in each step            
            ops = [merged, grad_step, training_model.loss, training_model.prob, training_model.accuracy_up,
                   training_model.precision_up, training_model.recall_up, training_model.confusion_matrix_up, 
                   training_model.pred, update_it_global, it_global]
            metrics = [training_model.accuracy, 
                       training_model.precision, 
                       training_model.recall, 
                       training_model.confusion_matrix,
                       training_model.accuracy_per_class]
            step = 0
            end_of_data = False
            while not end_of_data:
                try:
                    data_train = sess.run(train_data_gen.get_next_batch())
                    if(model_name == 'LSTM'):
                        inputs = {dyn_learning_rate : learning_rate,
                                  input_data : data_train[0][0],
                                  input_labels: data_train[0][1],
                                  input_seq_length : data_train[1]}
                    else:
                        inputs = {dyn_learning_rate : learning_rate,
                                  input_data : data_train[0],
                                  input_labels: data_train[1]}
    
                    run_meta = tf.RunMetadata()
                    # runs the optimization and updates the metrics
                    results = sess.run(ops, feed_dict=inputs, 
                                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                       run_metadata=run_meta)
                    profiler.add_step(step, run_meta)
                    
    #                option_builder = tf.profiler.ProfileOptionBuilder
    #                
    #                opts = (option_builder(option_builder.time_and_memory()).
    #                    with_step(-1). # with -1, should compute the average of all registered steps.
    #                    with_file_output('test.txt').
    #                    select(['micros','bytes','occurrence']).order_by('bytes').
    #                    build())
    #                # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
    #                profiler.profile_operations(options=opts)
                    # computes the metrics
                    metrics_ = sess.run(metrics)
                    # updates the learning rate and the old_loss
                    if(step % 100 == 0 and step > 0):
                        learning_rate /= 2
                    # plot the variables in TensorBoard
                    train_writer.add_summary(results[0], global_step=results[10])
                    # plot in console the metrics we want and hyperparameters
                    print('Epoch {}, step {}, accuracy : {}, loss : {}, learning_rate : {}'.format(epochs, step,
                          metrics_[0], results[2], learning_rate))
                    # save the weigths
                    if(results[10] % checkpoint_iter == 0):
                        saver.save(sess, os.path.join(checkpoint_dir,'training_{}.ckpt'.format(model_name)), it_global)
                        print('Checkpoint saved')
                    step += 1
                except tf.errors.OutOfRangeError:
                    if(epochs == num_epochs - 1):
                        saver.save(sess, os.path.join(checkpoint_dir,'training_{}.ckpt'.format(model_name)), it_global)
                        print('Last checkpoint saved')
                    end_of_data = True

            
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
    with G.as_default(), tf.device('/cpu:0'):        
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
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    start = time.time()
    with sess.as_default():
        if(restore_checkpoint(sess, saver, checkpoint_dir)):
            # Do not forget to reset metrics before using it !
            sess.run(local_init)
            sess.run(testing_model.reset_metrics())
            # Load the first batch of data in memory
            features, labels, metadata = test_data_gen.gen_batch_dataset(save_extracted_data = (model_name=='SF'), 
                                                                         retrieve_data = (model_name=='SF'),
                                                                         take_random_files = False,
                                                                         get_metadata=True)
            batch_it = 0
            while(features is not None and len(features) > 0):
                # Create the TF input pipeline and preprocess the data
                test_data_gen.create_tf_dataset_and_preprocessing(features, labels, metadata)
                # Get the first local batch 
                data_test = sess.run(test_data_gen.get_next_batch())
                # Testing loop 
                end_of_data = False
                
                while(not end_of_data):    
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
                    try:
                        data_test = sess.run(test_data_gen.get_next_batch())
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
                
                # Updates the data and the batch counter
                features, labels, metadata = test_data_gen.gen_batch_dataset(save_extracted_data = (model_name=='SF'), 
                                                                   retrieve_data = (model_name=='SF'),
                                                                   take_random_files = False, 
                                                                   get_metadata=True)
                batch_it += 1
        else:
            print('Impossible to test the model.')
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return testing_model

    
if __name__ == '__main__':
    tf.reset_default_graph()
    data = 'SF_LSTM' # in {'SF', 'SF_LSTM', 'MNIST', 'CIFAR-10'}
    test_model(data, True, True)
    #test_model(data, save_features=True)

