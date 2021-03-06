import tensorflow as tf
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, ConvLSTM2D, LSTM

class Model:
    
    name = None
    nb_classes = None
    training_mode = None
    dropout_prob = None
    batch_norm = None
    loss_weights = None
    lambda_reg = None
    model_built = None
    pb_kind = None
    
    def __init__(self, name = 'neural_network',
                 pb_kind = 'classification',
                 nb_classes = 2, training_mode = True,
                 batch_norm = True, dropout_prob = 0.4,
                 lambda_reg = 0.1,
                 regress_threshold = None,
                 loss_weights = [1, 1]):
        
        self.name = name
        self.pb_kind = pb_kind
        self.nb_classes = nb_classes
        self.training_mode = training_mode
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm
        self.loss_weights = loss_weights
        self.lambda_reg = lambda_reg
        self.regress_threshold = regress_threshold
            
            
    #                 GRAPH CONSTRUCTION                        #
    #############################################################
    
    # Uses Keras layers (for time ditribution)
    def build_lrcn(self, data):
         # data must have size n_vids x nb_frames x n_features 
         # * n_features = h*w*c+2 where (h,w,c) = (8,16,512) in the decoder output + 2 features = (size_h,size_w) of the init picture
         # * nb_frames = 12
         assert type(data) is tf.Tensor and len(data.shape) == 3 
         with tf.variable_scope(self.name):
            (_, nb_frames, n_features) = data.get_shape().as_list()
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5)
            self.input_layer = tf.cast(data, dtype=tf.float32)
            ### conv3-LSTM - 512 
            #self.convLSTM = ConvLSTM2D(filters=512, kernel_size=3, kernel_initializer=init, 
            #                           padding='same', activation='tanh', return_sequences=False, 
            #                           return_state=False, dropout=self.dropout_prob)(self.input_layer)
            self.LSTM = LSTM(units=512, input_shape=(nb_frames, n_features), activation='tanh', kernel_initializer=init)(self.input_layer)
            ### OPTION 1: SPP 
            #self.spp = self.spp_layer(self.convLSTM, [[4,4], [2,2], [1,1]], 'spp', pooling='TV')
            ### OPTION 2: NO SPP
            self.fc1 = Dense(256, activation='relu', kernel_initializer=init)(self.LSTM)
            if(self.training_mode): self.fc1 = Dropout(self.dropout_prob)(self.fc1)
            self.fc2 = Dense(128, activation='relu', kernel_initializer=init)(self.fc1)
            if(self.training_mode): self.fc2 = Dropout(self.dropout_prob)(self.fc2)

            if(self.pb_kind == 'classification'):
               self.output = Dense(self.nb_classes, kernel_initializer=init)(self.fc2)
            elif(self.pb_kind == 'regression'):
                self.output = tf.squeeze(Dense(1)(self.fc2), axis=1) # [[1.2], [2.3], ...] => [1.2, 2.3, ...]
            else:
                print('Illegal kind of problem for LRCN model: {}'.format(self.pb_kind))
                
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
            self.spp = self.spp_layer(self.pool5, [4, 2, 1], 'spp', pooling='TV')
            ### FC-1024
            self.dense1 = Dense(1024, activation='relu')(self.spp)
            if(self.training_mode):
                self.dense1 = Dropout(self.dropout_prob)(self.dense1)
            ### FC-1024
            self.dense2 = Dense(1024)(self.dense1)
            if(self.training_mode):
                self.dense2 = Dropout(self.dropout_prob)(self.dense2)
            ### FC-32 
            self.dense3 = Dense(32, activation='relu')(self.dense2)
            if(self.training_mode):
                self.dense3 = Dropout(self.dropout_prob)(self.dense3)
            ### output
            if(self.pb_kind == 'classification'):
                self.output = Dense(self.nb_classes)(self.dense3)
            elif(self.pb_kind == 'regression'):
                self.output = Dense(1)(self.dense3)
            else:
                print('Illegal kind of problem for VGG-16 model: {}'.format(self.pb_kind))
            
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
                                            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                            bias_initializer=tf.constant_initializer(0),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv1_1')
            if(self.batch_norm): self.conv1_1 = tf.layers.batch_normalization(self.conv1_1, training=self.training_mode, name='bn1_1')
            self.pool1 = tf.layers.max_pooling2d(self.conv1_1, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 128
            self.conv2_1 = tf.layers.conv2d(self.pool1, filters=128, kernel_size=(3, 3), 
                                            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                            bias_initializer=tf.constant_initializer(0), 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv2_1')
            if(self.batch_norm): self.conv2_1 = tf.layers.batch_normalization(self.conv2_1, training=self.training_mode, name='bn2_1')
            self.pool2 = tf.layers.max_pooling2d(self.conv2_1, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 256
            self.conv3_1 = tf.layers.conv2d(self.pool2, filters=256, kernel_size=(3, 3), 
                                            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                            bias_initializer=tf.constant_initializer(0),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv3_1')
            if(self.batch_norm): self.conv3_1 = tf.layers.batch_normalization(self.conv3_1, training=self.training_mode, name='bn3_1')
            self.pool3 = tf.layers.max_pooling2d(self.conv3_1, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 512
            self.conv4_1 = tf.layers.conv2d(self.pool3, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                            bias_initializer=tf.constant_initializer(0),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv4_1')
            if(self.batch_norm): self.conv4_1 = tf.layers.batch_normalization(self.conv4_1, training=self.training_mode, name='bn4_1')
            self.pool4 = tf.layers.max_pooling2d(self.conv4_1, pool_size=(2,2), strides=(2,2), padding='same')
            ### conv3 - 512
            self.conv5_1 = tf.layers.conv2d(self.pool4, filters=512, kernel_size=(3, 3), 
                                            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                            bias_initializer=tf.constant_initializer(0),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                            strides=(1,1), padding='same', activation='relu', name='conv5_1')
            if(self.batch_norm): self.conv5_1 = tf.layers.batch_normalization(self.conv5_1, training=self.training_mode, name='bn5_1')
            
            # OPTION 1 (with resizing/spp)
            self.pool5 = tf.image.resize_images(self.conv5_1, [8, 16], align_corners=True)
            
            # OPTION 2 (without resizing)
            #self.pool5 = tf.layers.max_pooling2d(self.conv5_1, pool_size=(2,2), strides=(2,2), padding='same')

            # Symmetric decoder
            ### unconv3 - 512
            self.unpool5 = tf.image.resize_images(self.pool5, tf.shape(self.conv5_1)[1:3], align_corners=True)
            self.unconv5_3 = tf.layers.conv2d(self.unpool5, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv5_3')
            if(self.batch_norm): self.unconv5_3 = tf.layers.batch_normalization(self.unconv5_3, training=self.training_mode, name='ubn5_3')
            ### unconv3 - 512
            self.unpool4 = tf.image.resize_images(self.unconv5_3, tf.shape(self.conv4_1)[1:3], align_corners=True)
            self.unconv4_3 = tf.layers.conv2d(self.unpool4, filters=512, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv4_3')
            if(self.batch_norm): self.unconv4_3 = tf.layers.batch_normalization(self.unconv4_3, training=self.training_mode, name='ubn4_3')
            ### unconv3 - 256
            self.unpool3 = tf.image.resize_images(self.unconv4_3, tf.shape(self.conv3_1)[1:3], align_corners=True)
            self.unconv3_3 = tf.layers.conv2d(self.unpool3, filters=256, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv3_3')
            if(self.batch_norm): self.unconv3_3 = tf.layers.batch_normalization(self.unconv3_3, training=self.training_mode, name='ubn3_3')
            ### unconv3 - 128
            self.unpool2 = tf.image.resize_images(self.unconv3_3, tf.shape(self.conv2_1)[1:3], align_corners=True)
            self.unconv2_2 = tf.layers.conv2d(self.unpool2, filters=128, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv2_2')
            if(self.batch_norm): self.unconv2_2 = tf.layers.batch_normalization(self.unconv2_2, training=self.training_mode, name='ubn2_2')
            ### unconv3 - 64
            self.unpool1 = tf.image.resize_images(self.unconv2_2, tf.shape(self.conv1_1)[1:3], align_corners=True)
            self.unconv1_2 = tf.layers.conv2d(self.unpool1, filters=64, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation='relu', name='unconv1_2')
            if(self.batch_norm): self.unconv1_2 = tf.layers.batch_normalization(self.unconv1_2, training=self.training_mode, name='ubn1_2')
            self.unconv1_1 = tf.layers.conv2d(self.unconv1_2, filters=nb_channels, kernel_size=(3, 3), 
                                              kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
                                              bias_initializer=tf.constant_initializer(0),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambda_reg),
                                              strides=(1,1), padding='same', activation=None, name='unconv1_1')
            
            # FC-layers for predicting the total variation
#            self.fc1 = tf.layers.dense(tf.layers.flatten(self.pool5), 256, activation='relu',
#                                       kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
#                                       bias_initializer=tf.constant_initializer(0), name='fc1')
#            self.fc2 = tf.layers.dense(self.fc1, 1, kernel_initializer=tf.initializers.random_normal(mean=0, stddev=1e-3), 
#                                       bias_initializer=tf.constant_initializer(0), name='fc2')
            
            if(self.pb_kind == 'encoder'):
                self.output = tf.cast(self.unconv1_1, dtype=tf.float32, name='output')
#                self.output_1 = tf.squeeze(self.fc2, axis=1, name='output_1')
            else:
                print('Illegal kind of problem for VGG-16 autoencoder model: {}'.format(self.pb_kind))
                
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
            self.dense = Dense(512, activation='relu')(self.dense)
            if(self.training):
                self.dense = Dropout(self.dropout_prob)(self.dense)
                
            if(self.pb_kind == 'classification'):
                self.output = Dense(self.nb_classes)(self.dense)
            elif(self.pb_kind == 'regression'):
                self.output = Dense(self.nb_classes)(self.dense3)
            else:
                print('Illegal kind of problem for LSTM model: {}'.format(self.pb_kind))
            
            
            self.model_built = 'LSTM'
            return self.output
    
    # argmax should have the same size as input_. Each indices i in argmax are computed according to:
    # i = c + C*(x + W*(y + H*b)) if the max value is initially at [b, y, x, c].
    @staticmethod
    def unpool_layer(input_, argmax, output_shape, name='unpool_layer'):
        with tf.variable_scope(name):
            input_unpool_flatten = tf.sparse_to_dense(tf.reshape(argmax, [-1]),
                                                      [tf.reduce_prod(output_shape)], 
                                                      tf.reshape(input_, [-1]), 
                                                      default_value=0, 
                                                      validate_indices=False)
            out = tf.reshape(input_unpool_flatten, output_shape)
        return out

    @staticmethod
    def spp_layer(input_, levels=[[4,4], [2,2], [1,1]], pooling='MAX', concatenate=True, return_argmax=False, name='spp_layer'): # pooling in {'AVG', 'MAX', 'TV'}
        # Input shape must be: b x h x w x c
        # Returns tensor of shape :
        #  * b x N  where N = prod(levels.*levels)*c if conca == True
        #  * b x levels[0] x levels[1] x c if conca == False
        if(return_argmax):
            print('Impossible to return argmax. Not Implemented yet.')
            raise
        with tf.variable_scope(name):
            shape = tf.cast(tf.shape(input_), tf.float32)
            if(not concatenate):
                assert len(levels) == 1
            pool_outputs = []
            for l in levels:
                # Compute the pooling manually by slicing the input tensor
                pool_size = tf.cast([tf.ceil(tf.div(shape[-3],l[0])), tf.ceil(tf.div(shape[-2], l[1]))], tf.int32)
                strides= tf.cast([tf.floordiv(shape[-3], l[0]), tf.floordiv(shape[-2], l[1])], tf.int32)
                for i in range(l[0]):
                    if(not concatenate):
                        pool_outputs += [[]]
                    for j in range(l[1]):
                        # bin (i,j)
                        tensor_slice = input_[:, i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1],:]
                        if(pooling == 'AVG'):
                            reduced_tensor = tf.reduce_mean(tensor_slice, axis=[1,2])
                        elif(pooling == 'MAX'):
                            reduced_tensor = tf.reduce_max(tensor_slice, axis=[1,2])
                        elif(pooling == 'TV'):
                            reduced_tensor = tf.reduce_sum(tf.abs(tensor_slice[:, 1:, :, :] - tensor_slice[:, :-1, :, :]), axis=[1,2]) +\
                                             tf.reduce_sum(tf.abs(tensor_slice[:, :, 1:, :] - tensor_slice[:, :, :-1, :]), axis=[1,2])
                        if(concatenate):
                            pool_outputs.append(reduced_tensor)
                        else:
                            pool_outputs[i] += [reduced_tensor]
            if(concatenate):
                spp = tf.concat(pool_outputs, 1)
            else:
                spp = tf.transpose(tf.convert_to_tensor(pool_outputs), perm=[1, 2, 0, 3]) # convert h' x w' x b x c to b x h' x w' x c
            return spp
    
    def update_confusion_matrix(self, labels, pred, name, nb_classes = None):
        if(nb_classes is None):
            nb_classes = self.nb_classes
        with tf.variable_scope(name):
            confusion_matrix= tf.get_variable('matrix', initializer=tf.zeros(shape=[nb_classes, nb_classes], dtype=tf.int32), trainable=False)
            update = tf.assign_add(confusion_matrix, tf.confusion_matrix(labels, pred, nb_classes), name='update')
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
            
            if(self.pb_kind == 'encoder'):
                self.TV = tf.divide(tf.reduce_sum(tf.abs(self.input_layer[:, 1:, :, :] - self.input_layer[:, :-1, :, :]), axis=[1,2,3])+\
                                    tf.reduce_sum(tf.abs(self.input_layer[:, :, 1:, :] - self.input_layer[:, :, :-1, :]), axis=[1,2,3]),
                                    tf.reduce_prod(tf.cast(tf.shape(self.input_layer)[1:], tf.float32)))
                
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input_layer, self.output))) #+ \
#                            tf.reduce_mean(tf.abs(tf.subtract(self.TV, self.output_1)))
                self.MSE, self.MSE_up = tf.metrics.mean_squared_error(self.TV, self.TV)#self.output_1, name="MSE")
                tf.summary.scalar('Loss', self.loss)
                tf.summary.scalar('MSE', self.MSE)
            elif(self.pb_kind == 'classification'):
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
            
            elif(self.pb_kind == 'regression'):
                self.loss = tf.reduce_mean(tf.abs(tf.subtract(labels, self.output)))
                self.MSE, self.MSE_up = tf.metrics.mean_squared_error(labels, self.output, name="MSE")
                if(self.regress_threshold is not None):
                    self.confusion_matrix, self.confusion_matrix_up = self.update_confusion_matrix(tf.to_int32(labels>=self.regress_threshold), 
                                                                                                   tf.to_int32(self.output>=self.regress_threshold), 
                                                                                                   name="confusion_matrix", nb_classes = 2)
                else:
                    self.confusion_matrix, self.confusion_matrix_up = [], []
                tf.summary.scalar('Loss', self.loss)
                
            else:
                print('Unknown problem : {}'.format(self.pb_kind))
                raise
            
    # Useful for testing phase
    def reset_metrics(self):
        with tf.variable_scope(self.name):
            if(self.pb_kind == 'classification' or self.pb_kind == 'regression'):
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

    
