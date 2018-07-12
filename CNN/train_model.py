import tensorflow as tf
import time, os, math
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
        return False


def train_model(config, train_data_gen):
    checkpoint_dir = config['checkpoint']
    tensorboard_dir= config['tensorboard']
    learning_rate = config['learning_rate'] # initial learning rate
    epsilon = config['tolerance'] # useful for updating learning rate
    batch_norm = config['batch_norm']
    nb_classes = config['nb_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_steps_per_epoch = config['num_steps']
    checkpoint_iter = config['checkpoint_iter']
    # Graph construction
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        input_data = tf.placeholder(dtype=tf.float32,\
                              shape=[batch_size] + train_data_gen.data_dims,\
                              name='data')
        input_labels = tf.placeholder(dtype=tf.int32,\
                                shape=[batch_size],\
                                name='labels')
        dyn_learning_rate = tf.placeholder(dtype=tf.float32,\
                                    shape=[],\
                                    name='learning_rate')
        training_model = model.Model('VGG_16', nb_classes, batch_norm=batch_norm)
        training_model.build_vgg16_like(input_data, input_labels)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=dyn_learning_rate)
            grads = optimizer.compute_gradients(training_model.loss)
            grad_step = optimizer.apply_gradients(grads)
        
        global_init = tf.global_variables_initializer() # Every weights
        local_init = tf.local_variables_initializer() # For accuracy
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

    # init and run training session
    print('Initializing training graph.')
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    sess.run(global_init)
    sess.run(local_init)
    start = time.time()
    with sess.as_default():
        utils.restore_checkpoint(sess, saver, checkpoint_dir)
        train_writer = tf.summary.FileWriter(tensorboard_dir+'/train', sess.graph)
        # training loop 
        it_global = 0
        old_loss = math.inf
        for epochs in range(num_epochs):
            # Load in memory the data
            x_train, y_train = train_data_gen.next_batch()
            # If we do not have data anymore, exit.
            if(x_train is None or y_train is None):
                break
            # Create a batch of batch_size 
            for step in range(num_steps_per_epoch):
                input_data_batch, input_labels_batch = data_gen.Data_Gen.get_random_batch(x_train, y_train, batch_size)
                inputs = {input_data : input_data_batch, input_labels : input_labels_batch, dyn_learning_rate : learning_rate}
                ops = [merged, grad_step, training_model.loss, training_model.accuracy, training_model.update_acc, training_model.prob]
                # run the optimization
                results = sess.run(ops, feed_dict=inputs)
                it_global += 1
                # upload the learning rate and the old_loss
                if(abs(results[3] - old_loss) < epsilon):
                    learning_rate /= 10
                old_loss = results[3]
                # plot the variables in TensorBoard
                train_writer.add_summary(results[0])
                # plot in console
                print('Iteration ({}) {}, accuracy : {}, loss : {}'.format(epochs, step, results[3], results[2]))
                # save the weigths
                if(it_global % checkpoint_iter == 0 or (step == num_steps_per_epoch-1 and epochs == num_epochs-1)):
                    saver.save(sess, save_path=os.path.join(checkpoint_dir,'.{}'.format(it_global)))
                    print('Checkpoint saved')
    
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    
    
    
def main_py():
    data = 'SF' # in {'SF', 'MNIST'}
    train_data_gen = data_gen.Data_Gen(data, utils.config[data]) # Memory size / batch= 4GB
    train_model(utils.config[data], train_data_gen)



