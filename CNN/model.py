import tensorflow as tf
import numpy as np

class Model:
    
    name = None
    nb_classes = None
    training_mode = None
    dropout_prob = None
    batch_norm = None
    
    
    def __init__(self, name = 'ccn_network',\
                 nb_classes = 2, training_mode = True,\
                 batch_norm = True, dropout_prob = 0.2):
        self.name = name
        self.nb_classes = nb_classes
        self.training_mode = training_mode
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
    
    #                 GRAPH CONSTRUCTION                        #
    #############################################################
    def build_vgg16_like(self, data, labels):
        # data must have size nb_pictures x height x width x nb_channels
        assert type(data) is tf.Tensor and len(data.shape) == 4
        assert type(labels) is tf.Tensor and labels.shape == (data.shape[0],)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            (nb_pics, height, width, nb_channels) = data.get_shape().as_list()
            
            self.input_layer = tf.reshape(data, [nb_pics, height, width, nb_channels])
            
            # CONV3-64 [h, w, c] --> [h, w, 64]
            self.conv1_1 = self.conv_layer(self.input_layer, [3, 3, nb_channels, 64], 'conv1_1')

            # CONV3-64 [h, w, 64] --> [h, w, 64]
            self.conv1_2 = self.conv_layer(self.conv1_1, [3, 3, 64, 64], 'conv1_2')
            
            # MAX POOLING [h, w, 64] --> [h/2, w/2, 64]
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1_2, name='pool1', pool_size=2, strides=2)
            
            ################################
            # CONV3-128 [h/2, w/2, 64] --> [h/2, w/2, 128]
            self.conv2_1 = self.conv_layer(self.pool1, [3, 3, 64, 128], 'conv2_1')

            # CONV3-128 [h/2, w/2, 128] --> [h/2, w/2, 128]
            self.conv2_2 = self.conv_layer(self.conv2_1, [3, 3, 128, 128], 'conv2_2')
     
            # MAX POOLING [h/2, w/2, 128] --> [h/4, w/4, 128]
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2_2, name='pool2', pool_size=2, strides=2)
            
            ################################
            # CONV3-256 [h/4, w/4, 128] --> [h/4, w/4, 256]
            self.conv3_1 = self.conv_layer(self.pool2, [3, 3, 128, 256], 'conv3_1')
            
            # CONV3-256 [h/4, w/4, 256] --> [h/4, w/4, 256]
            self.conv3_2 = self.conv_layer(self.conv3_1, [3, 3, 256, 256], 'conv3_2')
            
            # CONV3-256 [h/4, w/4, 256] --> [h/4, w/4, 256]
            self.conv3_3 = self.conv_layer(self.conv3_2, [3, 3, 256, 256], 'conv3_3')

            # MAX POOLING [h/4, w/4, 256] --> [h/8, w/8, 256]
            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3_3, name='pool3', pool_size=2, strides=2)
            
            ################################
            # CONV3-512 [h/8, w/8, 256] --> [h/8, w/8, 512]
            self.conv4_1 = self.conv_layer(self.pool3, [3, 3, 256, 512], 'conv4_1')
            
            # CONV3-512 [h/8, w/8, 512] --> [h/8, w/8, 512]
            self.conv4_2 = self.conv_layer(self.conv4_1, [3, 3, 512, 512], 'conv4_2')
            
            # CONV3-512 [h/8, w/8, 512] --> [h/8, w/8, 512]
            self.conv4_3 = self.conv_layer(self.conv4_2, [3, 3, 512, 512], 'conv4_3')

            # MAX POOLING [h/8, w/8, 512] --> [h/16, w/16, 512]
            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4_3, name='pool4', pool_size=2, strides=2)
            
            ################################
             # CONV3-512 [h/16, w/16, 512] --> [h/16, w/16, 512]
            self.conv5_1 = self.conv_layer(self.pool4, [3, 3, 512, 512], 'conv5_1')
            
            # CONV3-512 [h/16, w/16, 512] --> [h/16, w/16, 512]
            self.conv5_2 = self.conv_layer(self.conv5_1, [3, 3, 512, 512], 'conv5_2')
            
            # CONV3-512 [h/16, w/16, 512] --> [h/16, w/16, 512]
            self.conv5_3 = self.conv_layer(self.conv5_2, [3, 3, 512, 512], 'conv5_3')

            # MAX POOLING [h/16, w/16, 512] --> [h/32, w/32, 512]
            self.pool5 = tf.layers.max_pooling2d(inputs=self.conv5_3, name='pool5', pool_size=2, strides=2)
            
            ################################
            # FC-1024 [h/32, w/32, 512] --> [h*w/2]
            self.pool5_flat = tf.layers.flatten(self.pool5, name='pool5_flat')
            self.dense1 = self.fc_layer(self.pool5_flat, int(height*width/2), 1024, 'fc1')
            
            # FC-1024 [1024] --> [1024]
            self.dense2 = self.fc_layer(self.dense1, 1024, 1024, 'fc2')
            
            # FC-32 [1024] --> [32]
            self.dense3 = self.fc_layer(self.dense2, 1024, 32, 'fc3')
            
            # Results
            self.logits = self.fc_layer(self.dense3, 32, self.nb_classes, 'logits', activation=None, dropout=False)
            self.prob = tf.nn.softmax(self.logits)
            self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, self.nb_classes), self.logits)
            self.accuracy, self.update_acc = tf.metrics.accuracy(tf.argmax(self.logits, axis=1), labels)
            
            # Summary for tensorboard visualization
            self.weights_summary(tf.get_variable('conv5_3/kernel',shape=[3,3,512,512]), 'last_conv_weights')
            self.weights_summary(self.dense3, 'last_fc_layer')
            self.prob_summary(nb_pics)
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)
            
            
        return {'proba':self.prob, \
                'loss':self.loss, \
                'accuracy':self.accuracy, \
                'acc_update':self.update_acc}
    

    def conv_layer(self, input_, filter_shape, name, padding='SAME', activation='relu', strides=[1,1,1,1]):
        with tf.variable_scope(name):
            kernel = tf.Variable(tf.random_normal(filter_shape, mean=0.0, stddev=0.5, dtype=tf.float32), 'kernel')
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
            weights = tf.Variable(tf.random_normal([nb_input, nb_output], dtype=tf.float32), 'weights')
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
    
    def prob_summary(self, nb_pics):
        with tf.variable_scope('prob_summary'):
            for i in range(nb_pics):
                for j in range(self.nb_classes):
                    tf.summary.scalar('prob_pic_{}_class_{}'.format(i,j), self.prob[i,j])
                
    
    def weights_summary(self, var, name):
        with tf.variable_scope(name):
            var_flatten = tf.layers.flatten(var)
            mean = tf.reduce_mean(var_flatten)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
    
    
        
    
    
    
    
    
    
    
    
    
    
