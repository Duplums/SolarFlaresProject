import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, LSTM, Dense, Dropout, BatchNormalization

class Model:
    
    name = None
    nb_classes = None
    training_mode = None
    dropout_prob = None
    batch_norm = None
    loss_weights = None
    lambda_reg = None
    weigths_init = None
    model_built = None
    
    def __init__(self, name = 'neural_network',\
                 nb_classes = 2, training_mode = True,\
                 batch_norm = True, dropout_prob = 0.4,\
                 lambda_reg = 0.1,\
                 loss_weights = [1, 1]):
        self.name = name
        self.nb_classes = nb_classes
        self.training_mode = training_mode
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.loss_weights = loss_weights
        self.lambda_reg = lambda_reg
        self.weights_init = {}

    def init_weights(self, path_to_weights):
        if(self.name in {'VGG_16', 'VGG_16_encoder_decoder', 'LRCN'}):
            for f in os.listdir(path_to_weights):
                weights_name = os.path.splitext(f)[0] # Erase the extension .npy
                self.weights_init[weights_name] = np.load(os.path.join(path_to_weights, f))
                self.weights_init[weights_name] = np.random.normal(0, 0.01, self.weights_init[weights_name].shape)
                print('Weights {} loaded.'.format(weights_name))
        else:
            raise RuntimeError('Imposible to load weights for mode {}'.format(self.name))
            
            
    #                 GRAPH CONSTRUCTION                        #
    #############################################################
    
    # Uses Keras layers (for time ditribution)
    def build_lrcn(self, data):
         # data must have size 1 x nb_frames x height x width x nb_channels [just 1 video at a time since the size can change]
         assert type(data) is tf.Tensor and len(data.shape) == 5
         with tf.variable_scope(self.name):
            (_, nb_frames, height, width, nb_channels) = data.get_shape().as_list()
            
         # VGG-16 encoder (stack)
            ### conv3 - 64
            self.input_layer = tf.cast(data, dtype=tf.float32)            
            self.conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(height, width, nb_channels))(self.input_layer)
            self.conv1_1 = BatchNormalization()(self.conv1_1)
            self.conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv1_1)
            self.conv1_2 = BatchNormalization()(self.conv1_2)
            self.pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv1_2)
            ### conv3 - 128
            self.conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool1)
            self.conv2_1 = BatchNormalization()(self.conv2_1)
            self.conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv2_1)
            self.conv2_2 = BatchNormalization()(self.conv2_2)
            self.pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv2_2)
            ### conv3 - 256
            self.conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool2)
            self.conv3_1 = BatchNormalization()(self.conv3_1)
            self.conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv3_1)
            self.conv3_2 = BatchNormalization()(self.conv3_2)
            self.conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv3_2)
            self.conv3_3 = BatchNormalization()(self.conv3_3)
            self.pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv3_3)
            ### conv3 - 512
            self.conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool3)
            self.conv4_1 = BatchNormalization()(self.conv4_1)
            self.conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv4_1)
            self.conv4_2 = BatchNormalization()(self.conv4_2)
            self.conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv4_2)
            self.conv4_3 = BatchNormalization()(self.conv4_3)
            self.pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv4_3)
            ### conv3 - 512
            self.conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool4)
            self.conv5_1 = BatchNormalization()(self.conv5_1)
            self.conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv5_1)
            self.conv5_2 = BatchNormalization()(self.conv5_2)
            self.conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv5_2)
            self.conv5_3 = BatchNormalization()(self.conv5_3)
            self.pool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv5_3)
          # SPP [4, 2, 1]
            self.spp = self.spp_layer(self.pool5, [4, 2, 1], 'spp')
          # LSTM 
            self.lstm = LSTM(units=512, return_sequences=False, return_state=False, dropout=self.dropout_prob)(self.spp)
            self.output = Dense(self.nb_classes)(self.lstm)
            
            self.model_built = 'LRCN'
            return self.output
            
            
            
    def build_vgg16_like(self, data):
        # data must have size nb_pictures x height x width x nb_channels
        assert type(data) is tf.Tensor and len(data.shape) == 4
        with tf.variable_scope(self.name):
            (nb_pics, height, width, nb_channels) = data.get_shape().as_list()
            
          # VGG-16 encoder
             ### conv3 - 64
            self.input_layer = tf.cast(data, dtype=tf.float32)            
            self.conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(height, width, nb_channels))(self.input_layer)
            self.conv1_1 = BatchNormalization()(self.conv1_1)
            self.conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv1_1)
            self.conv1_2 = BatchNormalization()(self.conv1_2)
            self.pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv1_2)
            ### conv3 - 128
            self.conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool1)
            self.conv2_1 = BatchNormalization()(self.conv2_1)
            self.conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv2_1)
            self.conv2_2 = BatchNormalization()(self.conv2_2)
            self.pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv2_2)
            ### conv3 - 256
            self.conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool2)
            self.conv3_1 = BatchNormalization()(self.conv3_1)
            self.conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv3_1)
            self.conv3_2 = BatchNormalization()(self.conv3_2)
            self.conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv3_2)
            self.conv3_3 = BatchNormalization()(self.conv3_3)
            self.pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv3_3)
            ### conv3 - 512
            self.conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool3)
            self.conv4_1 = BatchNormalization()(self.conv4_1)
            self.conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv4_1)
            self.conv4_2 = BatchNormalization()(self.conv4_2)
            self.conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv4_2)
            self.conv4_3 = BatchNormalization()(self.conv4_3)
            self.pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv4_3)
            ### conv3 - 512
            self.conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.pool4)
            self.conv5_1 = BatchNormalization()(self.conv5_1)
            self.conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv5_1)
            self.conv5_2 = BatchNormalization()(self.conv5_2)
            self.conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(self.conv5_2)
            self.conv5_3 = BatchNormalization()(self.conv5_3)
            self.pool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(self.conv5_3)
            
            ### SPP [4, 2, 1]
            self.spp = self.spp_layer(self.pool5, [4, 2, 1], 'spp')
            ### FC-1024
            self.dense1 = Dense(1024)(self.spp)
            if(self.training):
                self.dense1 = Dropout(self.dropout_prob)(self.dense1)
            ### FC-1024
            self.dense2 = Dense(1024)(self.dense1)
            if(self.training):
                self.dense2 = Dropout(self.dropout_prob)(self.dense2)
            ### FC-32 
            self.dense3 = Dense(32)(self.dense2)
            if(self.training):
                self.dense3 = Dropout(self.dropout_prob)(self.dense3)
            ### output
            self.output = Dense(self.nb_classes)(self.dense3)

            self.weights_summary(tf.get_variable('conv1_1/kernel',shape=[3,3,nb_channels,64]), 'first_conv_weights')
            #self.weights_summary(tf.get_variable('conv5_3/kernel',shape=[3,3,512,512]), 'last_conv_weights')
            #self.weights_summary(self.dense2, '1024_fc_layer')
            self.weights_summary(self.output, 'last_fc_layer')
            #self.prob_summary(nb_pics)
            self.model_built = 'VGG_16'
            return self.output
        
    def build_vgg16_encoder_decoder(self, data):
        # data must have size nb_pictures x height x width x nb_channels
        assert type(data) is tf.Tensor and len(data.shape) == 4
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            (nb_pics, height, width, nb_channels) = data.get_shape().as_list()
            
            # VGG-16 encoder
            ### conv3 - 64
            self.input_layer = tf.cast(data, dtype=tf.float32)            
            self.conv1_1 = tf.layers.conv2d(self.input_layer, filters=64, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv1_1_W'][:3,:,:,:]), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv1_1_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv1_1')
            if(self.batch_norm): self.conv1_1 = tf.layers.batch_normalization(self.conv1_1)
            self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=64, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv1_2_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv1_2_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv1_2 = tf.layers.batch_normalization(self.conv1_2)
            self.pool1 = tf.layers.max_pooling2d(self.conv1_2, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 128
            self.conv2_1 = tf.layers.conv2d(self.pool1, filters=128, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv2_1_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv2_1_b']), 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv2_1 = tf.layers.batch_normalization(self.conv2_1)
            self.conv2_2 = tf.layers.conv2d(self.conv2_1, filters=128, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv2_2_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv2_2_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv2_2 = tf.layers.batch_normalization(self.conv2_2)
            self.pool2 = tf.layers.max_pooling2d(self.conv2_2, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 256
            self.conv3_1 = tf.layers.conv2d(self.pool2, filters=256, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv3_1_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv3_1_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv3_1 = tf.layers.batch_normalization(self.conv3_1)
            self.conv3_2 = tf.layers.conv2d(self.conv3_1, filters=256, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv3_2_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv3_2_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv3_2 = tf.layers.batch_normalization(self.conv3_2)
            self.conv3_3 = tf.layers.conv2d(self.conv3_2, filters=256, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv3_3_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv3_3_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv3_3 = tf.layers.batch_normalization(self.conv3_3)
            self.pool3 = tf.layers.max_pooling2d(self.conv3_3, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 512
            self.conv4_1 = tf.layers.conv2d(self.pool3, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv4_1_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv4_1_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv4_1 = tf.layers.batch_normalization(self.conv4_1)
            self.conv4_2 = tf.layers.conv2d(self.conv4_1, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv4_2_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv4_2_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv4_2 = tf.layers.batch_normalization(self.conv4_2)
            self.conv4_3 = tf.layers.conv2d(self.conv4_2, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv4_3_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv4_3_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv4_3 = tf.layers.batch_normalization(self.conv4_3)
            self.pool4 = tf.layers.max_pooling2d(self.conv4_1, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 512
            self.conv5_1 = tf.layers.conv2d(self.pool4, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv5_1_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv5_1_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv5_1 = tf.layers.batch_normalization(self.conv5_1)
            self.conv5_2 = tf.layers.conv2d(self.conv5_1, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv5_2_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv5_2_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv5_2 = tf.layers.batch_normalization(self.conv5_2)
            self.conv5_3 = tf.layers.conv2d(self.conv5_2, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.constant_initializer(self.weights_init['conv5_3_W']), 
                                            bias_initializer=tf.constant_initializer(self.weights_init['conv5_3_b']),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.conv5_3 = tf.layers.batch_normalization(self.conv5_3)
            self.pool5 = tf.layers.max_pooling2d(self.conv5_1, pool_size=(2,2), strides=(2,2), padding='same')
            
            # Symmetric decoder
            ### unconv3 - 512
            self.unpool5 = tf.image.resize_images(self.pool5, tf.shape(self.conv5_1)[1:3], align_corners=True)
            self.unconv5_3 = tf.layers.conv2d(self.unpool5, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv5_3_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv5_3_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv5_3 = tf.layers.batch_normalization(self.unconv5_3)
            self.unconv5_2 = tf.layers.conv2d(self.unconv5_3, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv5_2_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv5_2_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv5_2 = tf.layers.batch_normalization(self.unconv5_2)
            self.unconv5_1 = tf.layers.conv2d(self.unconv5_2, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv5_1_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv5_1_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv5_1 = tf.layers.batch_normalization(self.unconv5_1)
            self.unconv5_1 = tf.add(self.unconv5_1, self.conv5_1)
            ### unconv3 - 512
            self.unpool4 = tf.image.resize_images(self.unconv5_1, tf.shape(self.conv4_1)[1:3], align_corners=True)
            self.unconv4_3 = tf.layers.conv2d(self.unpool4, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv4_3_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv4_3_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv4_3 = tf.layers.batch_normalization(self.unconv4_3)
            self.unconv4_2 = tf.layers.conv2d(self.unconv4_3, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv4_2_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv4_2_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv4_2 = tf.layers.batch_normalization(self.unconv4_2)
            self.unconv4_1 = tf.layers.conv2d(self.unconv4_2, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv4_1_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv4_1_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv4_1 = tf.layers.batch_normalization(self.unconv4_1)
            self.unconv4_1 = tf.add(self.unconv4_1, self.conv4_1)
            ### unconv3 - 256
            self.unpool3 = tf.image.resize_images(self.unconv4_1, tf.shape(self.conv3_1)[1:3], align_corners=True)
            self.unconv3_3 = tf.layers.conv2d(self.unpool3, filters=256, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv3_3_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv3_3_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv3_3 = tf.layers.batch_normalization(self.unconv3_3)
            self.unconv3_2 = tf.layers.conv2d(self.unconv3_3, filters=256, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv3_2_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv3_2_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv3_2 = tf.layers.batch_normalization(self.unconv3_2)
            self.unconv3_1 = tf.layers.conv2d(self.unconv3_2, filters=256, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv3_1_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv3_1_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv3_1 = tf.layers.batch_normalization(self.unconv3_1)
            self.unconv3_1 = tf.add(self.unconv3_1, self.conv3_1)
            ### unconv3 - 128
            self.unpool2 = tf.image.resize_images(self.unconv3_2, tf.shape(self.conv2_1)[1:3], align_corners=True)
            self.unconv2_2 = tf.layers.conv2d(self.unpool2, filters=128, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv2_2_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv2_2_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')
            if(self.batch_norm): self.unconv2_2 = tf.layers.batch_normalization(self.unconv2_2)
            self.unconv2_1 = tf.layers.conv2d(self.unconv2_2, filters=128, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv2_1_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv2_1_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu')    
            if(self.batch_norm): self.unconv2_1 = tf.layers.batch_normalization(self.unconv2_1)
            self.unconv2_1 = tf.add(self.unconv2_1, self.conv2_1)
            ### unconv3 - 64
            self.unpool1 = tf.image.resize_images(self.unconv2_1, tf.shape(self.conv1_1)[1:3], align_corners=True)
            self.unconv1_2 = tf.layers.conv2d(self.unpool1, filters=64, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv1_2_W']), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv1_2_b']),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv1_2')
            if(self.batch_norm): self.unconv1_2 = tf.layers.batch_normalization(self.unconv1_2)
            self.unconv1_1 = tf.layers.conv2d(self.unconv1_2, filters=nb_channels, kernel_size=(3, 3), 
                                              kernel_initializer=tf.constant_initializer(self.weights_init['conv1_1_W'][:3,:,:,:]), 
                                              bias_initializer=tf.constant_initializer(self.weights_init['conv1_1_b'][:3]),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation=None, name='unconv1_1')
            self.output = tf.cast(self.unconv1_1, dtype=tf.float32, name='output')
            
            self.weights_summary(tf.get_variable('conv1_1/kernel',shape=[3,3,nb_channels,64]), 'first_conv_weights')
            self.weights_summary(tf.get_variable('unconv1_1/kernel',shape=[3,3,64,nb_channels]), 'last_unconv_weights')
            self.weights_summary(tf.get_variable('unconv1_2/kernel',shape=[3,3,128, 64]), 'before_last_unconv_weights')
            
            self.model_built = 'VGG_16_encoder_decoder'  
            return self.output
        
        
        
    def build_lstm(self, data, seq_length):
        # Data must have shape nb_seqs x max_time x n_inputs
        assert type(data) is tf.Tensor and len(data.shape) == 3
        assert type(seq_length) is tf.Tensor and len(seq_length.shape) == 1
        with tf.variable_scope(self.name):
            self.input_layer = data
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
            self.dense = tf.gather_nd(rnn_output, indices)
            self.output = self.fc_layer(self.dense, 512, self.nb_classes, 'logits')
            self.model_built = 'LSTM'
            self.weights_summary(self.output, 'last_fc_layer')
            return self.output
        
    
    
    def spp_layer(self, input_, levels=[4, 2, 1], name='spp_layer', pooling='AVG'): # pooling in {'AVG', 'MAX'}
        # Input shape can be: b x h x w x c (CNN) or b x nb_frames x h x w x c (LRCN, b = 1 if one video at a time)
        # Returns shape b x N or b x nb_frames x N where N = prod(levels.*levels)*c
        
        shape = tf.cast(tf.shape(input_), tf.float32)
        with tf.variable_scope(name):
            pool_outputs = []
            for l in levels:
                # Compute the pooling manually by slicing the input tensor
                pool_size = tf.cast([tf.ceil(tf.div(shape[-3],l)), tf.ceil(tf.div(shape[-2], l))], tf.int32)
                strides= tf.cast([tf.floordiv(shape[-3], l), tf.floordiv(shape[-2], l)], tf.int32)
                for i in range(l):
                    for j in range(l):
                        # bin (i,j)
                        if(len(input_.get_shape()) == 4):
                            tensor_slice = input_[:, i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1],:]
                            reduced_axis = [1, 2]
                        else:
                            tensor_slice = input_[:, :, i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1],:]
                            reduced_axis = [2, 3]
                        if(pooling == 'AVG'):
                            pool_outputs.append(tf.reduce_mean(tensor_slice, axis=reduced_axis))
                        else:
                            pool_outputs.append(tf.reduce_max(tensor_slice, axis=reduced_axis))
            
            spp = tf.concat(pool_outputs, 1)
            return spp
    
    def fc_layer(self, input_, nb_input, nb_output, name, activation='relu', dropout = True):
        with tf.variable_scope(name):
            weights = tf.Variable(tf.random_normal([nb_input, nb_output], mean=0, stddev=0.5, dtype=tf.float32), 'weights')
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
    
    def construct_results(self, labels = None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # Quick overview of the input batch
            tf.summary.scalar('N_input', tf.shape(self.input_layer)[0])
            tf.summary.scalar('input_size_per_image', tf.reduce_prod(tf.shape(self.input_layer)[1:]))
            if(self.model_built in {'VGG_16_encoder_decoder'}):
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input_layer, self.output))) #+ 0.1*tf.losses.get_regularization_loss()
                tf.summary.scalar('Loss', self.loss)
            else:
                # Results
                self.prob = tf.nn.softmax(self.output)
                self.pred = tf.argmax(self.prob, axis=1)
                self.loss = tf.losses.softmax_cross_entropy(tf.multiply(tf.one_hot(labels, self.nb_classes), self.loss_weights), self.output)
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
            if(self.model_built in {'LSTM', 'VGG_16'}):
                reset =  [tf.variables_initializer([self.confusion_matrix])]
            else:
                reset = []
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
        
    
    
    
    
    
    
    
    
    
    
