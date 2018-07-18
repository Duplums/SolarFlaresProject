

# Test the model created during the training phase. 
def test_model(config, test_data_gen):
    checkpoint_dir = config['checkpoint']
    nb_classes = config['nb_classes']
    batch_size = config['batch_size']
    num_steps_per_epoch = config['num_steps']
    # Testing graph
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        input_data = tf.placeholder(dtype=tf.float32,\
                              shape=[batch_size] + test_data_gen.data_dims,\
                              name='data')
        input_labels = tf.placeholder(dtype=tf.int32,\
                                shape=[batch_size],\
                                name='labels')

        testing_model = model.Model('VGG_16_Test', nb_classes, False)
        testing_model.build_vgg16_like(input_data, input_labels)
        saver = tf.train.Saver()

    # Init the graph with the last checkpoint. 
    sess = tf.Session(graph=G)
    tf.train.start_queue_runners(sess=sess)
    start = time.time()
    with sess.as_default():
        if(restore_checkpoint(sess, saver, checkpoint_dir)):
            it_global = 0
            # Loads in memory the data and saves/retrieves the features.
            x_train, y_train = test_data_gen.next_batch(True, True)
            # Testing loop 
            while(x_train is not None or y_train is not None):
                # Create a batch of batch_size 
                data_it = data_gen.Data_Iterator(x_train, y_train, batch_size)
                while(data_it.is_next_batch())
                    input_data_batch, input_labels_batch = data_it.get_next_batch()
                    inputs = {input_data : input_data_batch, input_labels : input_labels_batch}
                    ops = [training_model.prob, training_model.accuracy_up,
                           training_model.precision_up, training_model.recall_up, training_model.confusion_matrix_up, 
                           training_model.pred]
                    metrics = [training_model.accuracy, training_model.precision, training_model.recall, training_model.confusion_matrix,
                               training_model.accuracy_per_class]
                    # runs the optimization and updates the metrics
                    results = sess.run(ops, feed_dict=inputs)
                    # computes the metrics
                    metrics = sess.run(metrics)
                    it_global += 1
                    # updates the learning rate and the old_loss
                    if(abs(results[2] - old_loss) < epsilon):
                        learning_rate /= 2
                    old_loss = results[2]
                    # plot the variables in TensorBoard
                    train_writer.add_summary(results[0], global_step=it_global)
                    # plot in console the metrics we want and hyperparameters
                    print('Epoch {}, it {}, accuracy : {}, loss : {}, learning_rate : {}'.format(epochs, step, 
                          metrics[0], results[2], learning_rate))
                    # save the weigths
                    if(it_global % checkpoint_iter == 0 or (step == num_steps_per_epoch-1 and epochs == num_epochs-1)):
                        saver.save(sess, os.path.join(checkpoint_dir,'training_vgg16'), it_global)
                        print('Checkpoint saved')
            else:
                print('Impossible to test the model.')
            
    end = time.time()
    print("Time usage: " + str(timedelta(seconds=int(round(end-start)))))
    return training_model

