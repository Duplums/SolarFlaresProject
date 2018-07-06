import tensorflow as tf
import time, os
from datetime import timedelta
import model, utils, data_gen

# config: dict containing every info for training mode
# train_data_gen: load in RAM the data when needed

def train_model(config, train_data_gen):
    checkpoint_dir = config['checkpoint']
    tensorboard_dir= config['tensorboard']
    learning_rate = config['learning_rate']
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
        
        training_model = model.Model('VGG_16', nb_classes, batch_norm=batch_norm)
        training_model.build_vgg16_like(input_data, input_labels)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(training_model.loss)
            grad_step = optimizer.apply_gradients(grads)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    # init and run training session
    print('Initializing training graph.')
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    sess.run(init)
    
    start = time.time()
    with sess.as_default():
        utils.restore_checkpoint(sess, saver, checkpoint_dir)
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        # training loop
        it_global = 0
        for epochs in range(num_epochs):
            # Load in memory the data
            x_train, y_train = train_data_gen.next_batch()
            # Create a batch of batch_size 
            for step in range(num_steps_per_epoch):
                input_data_batch, input_labels_batch = train_data_gen.get_random_batch(x_train, y_train, batch_size)
                inputs = {input_data : input_data_batch, input_labels : input_labels_batch}
                ops = [grad_step, training_model.loss, training_model.accuracy, training_model.update_acc, training_model.proba]
                # run the optimization
                results = sess.run(ops, feed_dict=inputs)
                it_global += 1
                # plot the variables in TensorBoard
                merge = tf.summary.merge_all()
                writer.add_summary(merge)
                
                # save the weigths
                if(it_global % checkpoint_iter == 0 or (step == num_steps_per_epoch-1 and epochs == num_epochs-1)):
                    saver.save(sess, save_path=os.path.join(checkpoint_dir,'.{}'.format(it_global)))
                    print('Checkpoint saved')
    
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    
    
    
def main():
    config = utils.config
    paths = utils.paths
    train_data_gen = data_gen.Data_Gen(config['data_dims'], paths, 4*1024) # Memory size / batch= 4GB
    train_model(config, train_data_gen)











