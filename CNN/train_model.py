import tensorflow as tf
import time, os, traceback, argparse
from datetime import timedelta
import model, utils, data_gen
import numpy as np
import tracemalloc

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
    #epsilon = config['tolerance'] # useful for updating learning rate
    nb_classes = config['nb_classes']
    loss_weights = config['loss_weights']
    num_epochs = config['num_epochs']
    #checkpoint_iter = config['checkpoint_iter']
    model_name = config['model']
   
    # Creation of the training graph
    G = tf.Graph()
    with G.as_default():
        
        # Input TF pipeline
        train_data_gen = data_gen.Data_Gen(data, config, max_pic_size=[3000,3000])
        train_data_gen.create_tf_dataset_and_preprocessing(use_metadata = False)
        
        
        dyn_learning_rate = tf.placeholder(dtype=tf.float32,
                                           shape=[],
                                           name='learning_rate')
        # input_data[0] == features (and input_data[1] == labels, if any)
        if(model_name == 'LSTM'):
            input_data, input_seq_length = train_data_gen.get_next_batch()
        else:
            input_data = train_data_gen.get_next_batch()        
        
        it_global = tf.Variable(tf.constant(0, shape=[], dtype=tf.int32), trainable=False)
        update_it_global = tf.assign_add(it_global, tf.shape(input_data[0])[0]) 
        if(model_name == 'VGG_16'):
            training_model = model.Model('VGG_16', nb_classes, batch_norm=config['batch_norm'], 
                                         dropout_prob=config['dropout_prob'], loss_weights=loss_weights)
            training_model.build_vgg16_like(input_data[0])
            training_model.construct_results(input_data[1])
        elif(model_name == 'LSTM'):
            training_model = model.Model('LSTM', nb_classes, loss_weights=loss_weights)
            training_model.build_lstm(input_data[0], input_seq_length)
            training_model.construct_results(input_data[1])
        elif(model_name == 'VGG_16_encoder_decoder'):
            training_model = model.Model('VGG_16_encoder_decoder')
            training_model.build_vgg16_encoder_decoder(input_data[0])
            training_model.construct_results()
        elif(model_name == 'small_encoder_decoder'):
            training_model = model.Model('small_encoder_decoder')
            training_model.build_small_encoder_decoder(input_data[0])
            training_model.construct_results()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=dyn_learning_rate)
            grads = optimizer.compute_gradients(training_model.loss)
            grad_step = optimizer.apply_gradients(grads)
        
        merged = tf.summary.merge_all()
        
        # Adds the gradient and up_metrics operators
        if(model_name in {'LSTM', 'VGG_16'}):
            ops = [merged, grad_step, training_model.loss, 
                   training_model.prob, training_model.accuracy_up,
                   training_model.precision_up, training_model.recall_up, 
                   training_model.confusion_matrix_up, 
                   training_model.pred]
        elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
            ops = [merged, grad_step, training_model.loss]
        
        # Adds the global iteration counter    
        ops += [update_it_global]
        if(model_name in {'LSTM', 'VGG_16'}):
            metrics_ops = [training_model.accuracy, 
                           training_model.precision, 
                           training_model.recall, 
                           training_model.confusion_matrix,
                           training_model.accuracy_per_class]
        else:
            metrics_ops = []
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
    with sess.as_default():
        restore_checkpoint(sess, saver, checkpoint_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)      
        learning_rate = config['learning_rate']
        for epoch in range(num_epochs):
            # Decreases the learning rate every 2 epochs
            if(epoch % 2 == 1):
                learning_rate = learning_rate/2
            
            # Re-init the files in the data loader queue after each epoch
            train_data_gen.init_paths_to_file()
            
            batch_it = 0
            end_of_batch = False
            while(not end_of_batch):
                # Generates the next batch of data and loads it in memory
                end_of_batch = train_data_gen.gen_batch_dataset(save_extracted_data=False, 
                                                 retrieve_data=False,
                                                 take_random_files = True,
                                                 get_metadata=False)
                
                # Initializes the iterator on the current batch 
                sess.run(train_data_gen.data_iterator.initializer)
                if(model_name == 'LSTM'):
                    sess.run(train_data_gen.seq_length_iterator.initializer)
                    
                # Begins to load the data into the input TF pipeline
                step = 0
                end_of_data = False
                while(not end_of_data):
                    try:
                        # Runs the optimization and updates the metrics
                        results = sess.run(ops, feed_dict={dyn_learning_rate : learning_rate})
                        
                        # Computes the metrics
                        metrics = sess.run(metrics_ops)
                        
                        # Plots the variables in TensorBoard
                        train_writer.add_summary(results[0], global_step=results[-1])
                        
                        # Plots in console the metrics we want and hyperparameters
                        if(model_name in {'LSTM', 'VGG_16'}):
                            print('Epoch {}, Batch {}, step {}, accuracy : {}, loss : {}, learning_rate : {}'.format(epoch, batch_it, step,
                                  metrics[0], results[2], learning_rate))
                        elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
                            print('Epoch {}, Batch {}, step {}, loss : {}, learning_rate : {}'.format(epoch, batch_it, step,
                                  results[2], learning_rate))
                      
                        step += 1
                    except tf.errors.OutOfRangeError:                            
                        end_of_data = True
                # Gets the global iteration counter
                global_counter = sess.run(it_global)
                
                # Saves the weights 
                saver.save(sess, os.path.join(checkpoint_dir,'training_{}.ckpt'.format(model_name)), global_counter) 
                batch_it += 1

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
    
    # Creation of the training graph
    G = tf.Graph()
    with G.as_default():
        # Input TF pipeline
        
        test_data_gen = data_gen.Data_Gen(data, config, training = test_on_training, 
                                           max_pic_size=[3000,3000])
        test_data_gen.create_tf_dataset_and_preprocessing(use_metadata = True)
        
        # input_data[0] == features (and input_data[1] == labels, if any)
        if(model_name == 'LSTM'):
            input_data, input_seq_length = test_data_gen.get_next_batch()
        else:
            input_data = test_data_gen.get_next_batch()        
        
        it_global = tf.Variable(tf.constant(0, shape=[], dtype=tf.int32), trainable=False)
        update_it_global = tf.assign_add(it_global, tf.shape(input_data[0])[0]) 
        if(model_name == 'VGG_16'):
            testing_model = model.Model('VGG_16', nb_classes, training_mode=False)
            testing_model.build_vgg16_like(input_data[0])
            testing_model.construct_results(input_data[1])
        elif(model_name == 'LSTM'):
            testing_model = model.Model('LSTM', nb_classes, training_mode=False)
            testing_model.build_lstm(input_data[0], input_seq_length)
            testing_model.construct_results(input_data[1])
        elif(model_name == 'VGG_16_encoder_decoder'):
            testing_model = model.Model('VGG_16_encoder_decoder', training_mode=False)
            testing_model.build_vgg16_encoder_decoder(input_data[0])
            testing_model.construct_results()
        elif(model_name == 'small_encoder_decoder'):
            testing_model = model.Model('small_encoder_decoder', training_mode=False)
            testing_model.build_small_encoder_decoder(input_data[0])
            testing_model.construct_results()
        
        # Adds the up_metrics operators
        if(model_name in {'LSTM', 'VGG_16'}):
            ops = [testing_model.accuracy_up,
                   testing_model.precision_up, 
                   testing_model.recall_up, 
                   testing_model.confusion_matrix_up, 
                   testing_model.pred]
        elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
            ops = [testing_model.loss]
        
        # Adds the global iteration counter    
        ops += [update_it_global]
        if(model_name in {'LSTM', 'VGG_16'}):
            metrics_ops = [testing_model.accuracy, 
                           testing_model.precision, 
                           testing_model.recall, 
                           testing_model.confusion_matrix,
                           testing_model.accuracy_per_class]
        else:
            metrics_ops = []
        
        # Selects the features that need to be saved
        if(save_features):
            if(model_name == 'VGG_16'):
                ops += [testing_model.dense2]
            elif(model_name == 'LSTM'):
                ops += [testing_model.output]
                
#   TODO LIST:                
#            elif(model_name == 'VGG_16_encoder_decoder'):
#                ops += [testing_model.pool5]
#            elif(model_name == 'small_encoder_decoder'):
#                ops += [testing_model.pool3]
        
        # We just want to initialize the metrics.
        local_init = [tf.local_variables_initializer(), it_global.initializer]
        saver = tf.train.Saver()

    # Init and run testing session
    print('Initializing testing graph.')
    sess = tf.Session(graph=G, config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        if(restore_checkpoint(sess, saver, checkpoint_dir)):
            # Initializes all metrics.
            sess.run(local_init)
            sess.run(testing_model.reset_metrics())  
            
            # Starts the test on all files
            batch_it = 0
            end_of_batch = False
            while(not end_of_batch):
                # Generates the next batch of data and loads it in memory
                end_of_batch = test_data_gen.gen_batch_dataset(save_extracted_data=False, 
                                                               retrieve_data=False,
                                                               take_random_files = False,
                                                               get_metadata=True)
                    
                # Initializes the iterator on the current batch 
                sess.run(test_data_gen.data_iterator.initializer)
                if(model_name == 'LSTM'):
                    sess.run(test_data_gen.seq_length_iterator.initializer)
                
                # Begins to load the data into the input TF pipeline
                step = 0
                end_of_data = False
                while(not end_of_data):
                    try:
                        # Updates the metrics according to the current batch
                        results = sess.run(ops)
                            
                        # Computes the metrics
                        metrics = sess.run(metrics_ops)
                        
                        # If necessary, save the features extracted in memory
                        if(save_features):
                            features = results[-1]
                            test_data_gen.add_output_features(features)

                        step += 1
                    except tf.errors.OutOfRangeError:                            
                        end_of_data = True
                
                # Gets the global iteration counter (for plots)
                global_counter = sess.run(it_global)
                
                # If necessary, dump the features extracted on disk
                if(save_features):
                    test_data_gen.dump_output_features()
                
                # Plot in the console the current results
                print('-----\nBATCH {} -----\n'.format(batch_it))
                print('Current counter: {}\n'.format(global_counter))
                if(model_name in {'LSTM', 'VGG_16'}):
                    print('Accuracy : {}, Precision : {} \nRecall : {}, Accuracy per class : {}\nConfusion Matrix : {}\n'.
                      format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
                elif(model_name in {'VGG_16_encoder_decoder', 'small_encoder_decoder'}):
                    print('Loss : {}'.format(results[0]))
                
                # Finally, saves the confusion matrix, if needed.
                if(model_name in {'LSTM', 'VGG_16'}):
                    if(test_on_training):
                        np.save(checkpoint_dir+'/training_confusion_matrix', metrics[4])
                    else:
                        np.save(checkpoint_dir+'/testing_confusion_matrix', metrics[4])
                
                
    
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
    tracemalloc.start()
    tf.reset_default_graph()
    if(args.testing):
        test_model(data, args.test_on_training, args.save_features)
    else:
        t = train_model(data)

