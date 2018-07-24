import tensorflow as tf
import time, os, math, traceback
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
    batch_norm = config['batch_norm']
    nb_classes = config['nb_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    checkpoint_iter = config['checkpoint_iter']
    # Training graph
    G = tf.Graph()
    train_data_gen = data_gen.Data_Gen(data, config, G, max_pic_size=[3000,3000])
    
    with G.as_default(), tf.device('/cpu:0'):
        
        dyn_learning_rate = tf.placeholder(dtype=tf.float32,
                                           shape=[],
                                           name='learning_rate')
        
        input_data = tf.placeholder(dtype=tf.float32,
                                    shape=[None]+config['data_dims'],
                                    name='data')
        input_labels = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name='labels')
        it_global = tf.Variable(tf.constant(0, shape=[], dtype=tf.int32), trainable=False)

        update_it_global = tf.assign_add(it_global, batch_size) 
        training_model = model.Model('VGG_16', nb_classes, batch_norm=batch_norm)
        training_model.build_vgg16_like(input_data, input_labels)
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
        #restore_checkpoint(sess, saver, checkpoint_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir+'/train', sess.graph)
        # training loop 
        old_loss = math.inf
        for epochs in range(num_epochs):
            # re-init the learning rate at the beginning of each epoch
            learning_rate = config['learning_rate']
            # Load the next batch of data in memory
            features, labels = train_data_gen.gen_batch_dataset(True, True)
            # Create the TF input pipeline and preprocess the data
            train_data_gen.create_tf_dataset_and_preprocessing(features, labels)
            # Get the first local batch 
            x_train, y_train = sess.run(train_data_gen.get_next_batch())
            # Computes every ops in each step            
            ops = [merged, grad_step, training_model.loss, training_model.prob, training_model.accuracy_up,
                   training_model.precision_up, training_model.recall_up, training_model.confusion_matrix_up, 
                   training_model.pred, update_it_global, it_global]
            metrics = [training_model.accuracy, training_model.precision, training_model.recall, training_model.confusion_matrix,
                       training_model.accuracy_per_class]
            step = 0
            while x_train is not None and y_train is not None:
                inputs = {dyn_learning_rate : learning_rate,
                          input_data : x_train,
                          input_labels: y_train}
                # runs the optimization and updates the metrics
                results = sess.run(ops, feed_dict=inputs)
                # computes the metrics
                metrics = sess.run(metrics)
                # updates the learning rate and the old_loss
                if(abs(results[2] - old_loss) < epsilon):
                    learning_rate /= 2
                old_loss = results[2]
                # plot the variables in TensorBoard
                train_writer.add_summary(results[0], global_step=results[10])
                # plot in console the metrics we want and hyperparameters
                print('Epoch {}, step {}, accuracy : {}, loss : {}, learning_rate : {}'.format(epochs, step,
                      metrics[0], results[2], learning_rate))
                step += 1
                x_train, y_train = sess.run(train_data_gen.get_next_batch())
                # save the weigths
                if(results[10] % checkpoint_iter == 0 or (epochs == num_epochs-1 and x_train is None)):
                    saver.save(sess, os.path.join(checkpoint_dir,'training_vgg16.ckpt'), it_global)
                    print('Checkpoint saved')
                
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return training_model

# Test the model created during the training phase. 
def test_model(config, test_data_gen):
    checkpoint_dir = config['checkpoint']
    nb_classes = config['nb_classes']
    batch_size = config['batch_size']
    # Testing graph
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        input_data = tf.placeholder(dtype=tf.float32,\
                              shape=[None] + test_data_gen.data_dims,\
                              name='data')
        input_labels = tf.placeholder(dtype=tf.int32,\
                                shape=[None],\
                                name='labels')

        testing_model = model.Model('VGG_16', nb_classes, False)
        testing_model.build_vgg16_like(input_data, input_labels)
        saver = tf.train.Saver()
        local_init = tf.local_variables_initializer()
        
    # Init the graph with the last checkpoint. 
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    start = time.time()
    with sess.as_default():
        if(restore_checkpoint(sess, saver, checkpoint_dir)):
            # Do not forget to reset metrics before using it !
            sess.run(local_init)
            sess.run(testing_model.reset_metrics())
            # Loads in memory the data and saves/retrieves the features.
            x_train, y_train = test_data_gen.next_batch(False, False)
            # Testing loop 
            it_global = 0
            while(x_train is not None or y_train is not None or
                  len(x_train) == 0 or len(y_train) == 0):
                # Create a batch of batch_size 
                data_it = data_gen.Data_Iterator(x_train, y_train, batch_size)
                it_batch = 0
                while(data_it.is_next_batch()):
                    print('Batch {}, scanning {}%'.format(it_global, data_it.where_is_it()*100))
                    input_data_batch, input_labels_batch = data_it.next_batch()
                    inputs = {input_data : input_data_batch, input_labels : input_labels_batch}
                    ops = [testing_model.accuracy_up, testing_model.precision_up,
                           testing_model.recall_up, testing_model.confusion_matrix_up]
                    metrics = [testing_model.accuracy, testing_model.precision, 
                               testing_model.recall, testing_model.accuracy_per_class,
                               testing_model.confusion_matrix]
                    # Update the metrics 
                    sess.run(ops, feed_dict=inputs)
                    it_batch += 1
                # computes the metrics
                metrics = sess.run(metrics)
                # plot in console the metrics we want and hyperparameters
                print('-----\nBATCH {} -----\n'.format(it_global))
                print('Accuracy : {}, Precision : {} \nRecall : {}, Accuracy per class : {}\nConfusion Matrix : {}\n-----'.format(metrics[0], 
                  metrics[1], metrics[2], metrics[3], metrics[4]))
                x_train, y_train = test_data_gen.next_batch(False, False)
                it_global += 1
        else:
            print('Impossible to test the model.')
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return testing_model

    
def main_py():
    tf.reset_default_graph()
    data = 'SF' # in {'SF', 'MNIST', 'CIFAR-10'}
    train_model(data)
    #test_data_gen = data_gen.Data_Gen(data, utils.config[data], training=False)
    #test_model(utils.config[data], test_data_gen)

main_py()

