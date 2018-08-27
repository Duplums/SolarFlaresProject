import tensorflow as tf
import numpy as np

class Model:
    
    name = None
    nb_classes = None
    training_mode = None
    dropout_prob = None
    batch_norm = None
    loss_weights = None
    
    def __init__(self, name = 'neural_network',\
                 nb_classes = 2, training_mode = True,\
                 batch_norm = True, dropout_prob = 0.2,\
                 loss_weights = [1, 1]):
        self.name = name
        self.nb_classes = nb_classes
        self.training_mode = training_mode
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.loss_weights = loss_weights
    
    #                 GRAPH CONSTRUCTION                        #
    #############################################################
    def build_vgg16_like(self, data):
        # data must have size nb_pictures x height x width x nb_channels
        assert type(data) is tf.Tensor and len(data.shape) == 4
        with tf.variable_scope(self.name):
            (nb_pics, height, width, nb_channels) = data.get_shape().as_list()
            
            self.input_layer = tf.cast(data, dtype=tf.float32)            
            # CONV3-64 [h, w, c] --> [h, w, 64]
            self.conv1_1 = self.conv_layer(self.input_layer, [3, 3, nb_channels, 64], 'conv1_1')

            # CONV3-64 [h, w, 64] --> [h, w, 64]
            self.conv1_2 = self.conv_layer(self.conv1_1, [3, 3, 64, 64], 'conv1_2')
            
            # MAX POOLING [h, w, 64] --> [h/2, w/2, 64]
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1_2, name='pool1', pool_size=2, strides=2, padding='same')
            
            ################################
            # CONV3-128 [h/2, w/2, 64] --> [h/2, w/2, 128]
            self.conv2_1 = self.conv_layer(self.pool1, [3, 3, 64, 128], 'conv2_1')

            # CONV3-128 [h/2, w/2, 128] --> [h/2, w/2, 128]
            self.conv2_2 = self.conv_layer(self.conv2_1, [3, 3, 128, 128], 'conv2_2')
     
            # MAX POOLING [h/2, w/2, 128] --> [h/4, w/4, 128]
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2_2, name='pool2', pool_size=2, strides=2, padding='same')
            
            ################################
            # CONV3-256 [h/4, w/4, 128] --> [h/4, w/4, 256]
            self.conv3_1 = self.conv_layer(self.pool2, [3, 3, 128, 256], 'conv3_1')
            
            # CONV3-256 [h/4, w/4, 256] --> [h/4, w/4, 256]
            self.conv3_2 = self.conv_layer(self.conv3_1, [3, 3, 256, 256], 'conv3_2')
            
            # CONV3-256 [h/4, w/4, 256] --> [h/4, w/4, 256]
            self.conv3_3 = self.conv_layer(self.conv3_2, [3, 3, 256, 256], 'conv3_3')

            # MAX POOLING [h/4, w/4, 256] --> [h/8, w/8, 256]
            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3_3, name='pool3', pool_size=2, strides=2, padding='same')
            
            ################################
            # CONV3-256 [h/8, w/8, 256] --> [h/8, w/8, 256]
            self.conv4_1 = self.conv_layer(self.pool3, [3, 3, 256, 256], 'conv4_1')
            
            # CONV3-256 [h/8, w/8, 256] --> [h/8, w/8, 256]
            self.conv4_2 = self.conv_layer(self.conv4_1, [3, 3, 256, 256], 'conv4_2')
            
            # CONV3-256 [h/8, w/8, 256]--> [h/8, w/8, 256]
            self.conv4_3 = self.conv_layer(self.conv4_2, [3, 3, 256, 256], 'conv4_3')

            # MAX POOLING [h/8, w/8, 256] --> [h/16, w/16, 256]
            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4_3, name='pool4', pool_size=2, strides=2, padding='same')
            
            ################################
            # CONV3-512 [h/16, w/16, 256] --> [h/16, w/16, 512]
            self.conv5_1 = self.conv_layer(self.pool4, [3, 3, 256, 512], 'conv5_1')
            
            # CONV3-512 [h/16, w/16, 512] --> [h/16, w/16, 512]
            self.conv5_2 = self.conv_layer(self.conv5_1, [3, 3, 512, 512], 'conv5_2')
            
            # CONV3-512 [h/16, w/16, 512] --> [h/16, w/16, 512]
            self.conv5_3 = self.conv_layer(self.conv5_2, [3, 3, 512, 512], 'conv5_3')

            # MAX POOLING [h/16, w/16, 512] --> [h/32, w/32, 512]
            self.pool5 = tf.layers.max_pooling2d(inputs=self.conv5_3, name='pool5', pool_size=2, strides=2, padding='same')
            
            ################################
            # SPATIAL PYRAMID POOLING [h/32, w/32, 512] --> [(4*4+2*2+1*1)*512]=[10752] for levels = [4, 2, 1]
            self.spp = self.spp_layer(self.pool5, [4, 2, 1], 'spp')
            
            ################################
            # FC-1024 [10752] --> [1024]
            self.dense1 = self.fc_layer(self.spp, 10752, 1024, 'fc1')
            
            # FC-1024 [1024] --> [1024]
            self.dense2 = self.fc_layer(self.dense1, 1024, 1024, 'fc2')
            
            # FC-32 [1024] --> [32]
            self.output = self.fc_layer(self.dense2, 1024, 32, 'fc3')
            self.logits = self.fc_layer(self.output, 32, self.nb_classes, 'logits', activation=None, dropout=False)

            self.weights_summary(tf.get_variable('conv1_1/kernel',shape=[3,3,nb_channels,64]), 'first_conv_weights')
            #self.weights_summary(tf.get_variable('conv5_3/kernel',shape=[3,3,512,512]), 'last_conv_weights')
            #self.weights_summary(self.dense2, '1024_fc_layer')
            self.weights_summary(self.output, 'last_fc_layer')
            #self.prob_summary(nb_pics)
            return self.output
            
    
    def build_lstm(self, data, seq_length):
        # Data must have shape nb_seqs x max_time x n_inputs
        assert type(data) is tf.Tensor and len(data.shape) == 3
        assert type(seq_length) is tf.Tensor and len(seq_length.shape) == 1
        with tf.variable_scope(self.name):
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units=512, use_peepholes=True, name='LSTM_Cell')
            rnn_output, rnn_last_state = tf.nn.dynamic_rnn(lstm_cell, data, seq_length, dtype=tf.float32)
            # rnn_output has size nb_seqs x max_time x output_cell_size [=512]
            # We're only interest in the last ouput for each sequence (defined according
            # to the seq_length)
            batch_range = tf.range(tf.shape(data)[0])
            # indices = [[0, seq_len[0]-1], [1, seq_len[1]-1], ..., [nb_seqs, seq_len[-1]-1]]
            indices = tf.stack([batch_range, seq_length - 1], axis=1)
            # in each sequence i (range), select the line seq_len[i]-1
            # size : nb_seqs x n_inputs
            self.output = tf.gather_nd(rnn_output, indices)
            self.logits = self.fc_layer(self.output, 512, self.nb_classes, 'logits', activation=None, dropout=False)
            return self.output
        
    
    
    def spp_layer(self, input_, levels=[4, 2, 1], name='spp_layer', pooling='AVG'): # pooling in {'AVG', 'MAX'}
        shape = tf.cast(tf.shape(input_), tf.float32)
        with tf.variable_scope(name):
            pool_outputs = []
            for l in levels:
                # Compute the pooling manually by slicing the input tensor
                pool_size = tf.cast([tf.ceil(tf.div(shape[1],l)), tf.ceil(tf.div(shape[2], l))], tf.int32)
                strides= tf.cast([tf.floordiv(shape[1], l), tf.floordiv(shape[2], l)], tf.int32)
                for i in range(l):
                    for j in range(l):
                        # bin (i,j)
                        tensor_slice = input_[:, i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1],:]
                        if(pooling == 'AVG'):
                            pool_outputs.append(tf.reduce_mean(tensor_slice, axis=[1, 2]))
                        else:
                            pool_outputs.append(tf.reduce_max(tensor_slice, axis=[1,2]))
            
            spp = tf.concat(pool_outputs, 1)
            return spp

    def conv_layer(self, input_, filter_shape, name, padding='SAME', activation='relu', strides=[1,1,1,1]):
        with tf.variable_scope(name):
            kernel = tf.Variable(tf.random_normal(filter_shape, mean=1, stddev=0.5, dtype=tf.float32), 'kernel')
            biases = tf.Variable(tf.constant(0, shape=[filter_shape[-1]], dtype=tf.float32), 'biases')
            # convolve and add bias            
            conv = tf.nn.conv2d(input_, kernel, strides, padding)
            conv = tf.nn.bias_add(conv, biases)
            # if a batch norm is needed, apply it
            if(self.batch_norm):
                conv = tf.layers.batch_normalization(conv, training=self.training_mode)
            if(activation == 'relu'):
                conv = tf.nn.relu(conv)
            elif(activation == 'sigmoid'):
                conv = tf.nn.sigmoid(conv)
            elif(activation == 'tanh'):
                conv = tf.nn.tanh(conv)
            
            return conv
    
    def fc_layer(self, input_, nb_input, nb_output, name, activation='relu', dropout = True):
        with tf.variable_scope(name):
            weights = tf.Variable(tf.random_normal([nb_input, nb_output], mean=1, stddev=0.5, dtype=tf.float32), 'weights')
            biases = tf.Variable(tf.constant(0, shape=[nb_output], dtype=tf.float32), 'biases')
            # mult and add bias
            fc = tf.matmul(input_, weights) + biases
            # if a batch norm is needed, apply it
            if(self.batch_norm):
                fc = tf.layers.batch_normalization(fc, training=self.training_mode)
            if(activation == 'relu'):
                fc = tf.nn.relu(fc)
            elif(activation == 'sigmoid'):
                fc = tf.nn.signoid(fc)
            elif(activation == 'tanh'):
                fc = tf.nn.tanh(fc)
            if(dropout):
                fc = tf.layers.dropout(fc, self.dropout_prob, training=self.training_mode)
            
            return fc
    
    def update_confusion_matrix(self, labels, pred, name):
        with tf.variable_scope(name):
            confusion_matrix= tf.get_variable('matrix', initializer=tf.zeros(shape=[self.nb_classes, self.nb_classes], dtype=tf.int32), trainable=False)
            update = tf.assign_add(confusion_matrix, tf.confusion_matrix(labels, pred, self.nb_classes), name='update')
            return confusion_matrix, update
    
    @staticmethod
    def compute_acc_per_class(confusion_matrix):
        acc = tf.truediv(tf.diag_part(confusion_matrix), tf.reduce_sum(confusion_matrix, axis=1)) 
        return acc    
    
    def construct_results(self, labels):
        with tf.variable_scope(self.name):
            # Results
            self.prob = tf.nn.softmax(self.logits)
            self.pred = tf.argmax(self.prob, axis=1)
            self.loss = tf.losses.softmax_cross_entropy(tf.multiply(tf.one_hot(labels, self.nb_classes), self.loss_weights), self.logits)
            # Define our metrics
            self.accuracy, self.accuracy_up = tf.metrics.accuracy(labels, self.pred, name="accuracy")
            self.precision, self.precision_up = tf.metrics.precision(labels, self.pred, name="precision")
            self.recall, self.recall_up = tf.metrics.recall(labels, self.pred, name="recall")
            self.confusion_matrix, self.confusion_matrix_up = self.update_confusion_matrix(labels, self.pred, name="confusion_matrix")
            self.accuracy_per_class = self.compute_acc_per_class(self.confusion_matrix)
            # Summary for tensorboard visualization
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)
            tf.summary.scalar('Precision', self.precision)
            tf.summary.scalar('Recall', self.recall)
            confusion_image = tf.reshape( tf.cast(self.confusion_matrix, tf.float32),
                                             [1, self.nb_classes, self.nb_classes, 1])
            tf.summary.image('confusion', confusion_image)
            self.vector_summary(self.accuracy_per_class, 'Accuracy_Per_Class')
    
    # Useful for testing phase
    def reset_metrics(self):
        with tf.variable_scope(self.name):
            reset =  [tf.variables_initializer([self.confusion_matrix])]
            return(reset)
    
    def prob_summary(self, nb_pics):
        with tf.variable_scope('prob_summary'):
            for i in range(nb_pics):
                for j in range(self.nb_classes):
                    tf.summary.scalar('prob_pic_{}_class_{}'.format(i,j), self.prob[i,j])
                
    def weights_summary(self, var, name):
        with tf.variable_scope(name):
            var_flatten = tf.reshape(var, [-1])
            mean = tf.reduce_mean(var_flatten)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
    
    def vector_summary(self, vec, name):
        with tf.variable_scope(name):
            n_vec = vec.get_shape()[0]
            for k in range(n_vec):
                tf.summary.scalar('Component_{}'.format(k), vec[k])
        
    
    
    
    
    
    
    
    
    
    
