# 后续可以外部导入最佳网络结构


class BiLSTM_Model:
    def __init__(self, n_units):
        '''
        各种参数
        :param n_units: for bilstm

        '''
        self.n_units = n_units

    def get_emb_layer(self, emb_matrix, input_length, trainable):
        '''
        embedding层 index 从maxtrix 里 lookup出向量 
        '''
        embedding_dim = emb_matrix.shape[-1]
        input_dim = emb_matrix.shape[0]
        emb_layer = keras.layers.Embedding(input_dim, embedding_dim,
                                           input_length=input_length,
                                           weights=[emb_matrix],
                                           trainable=trainable)
        return emb_layer

    def get_input_layer(self, name=None, dtype="int32"):
        '''
        input层 字典索引序列
        '''
        input_layer = keras.Input(
            shape=(seq_length_creative_id,), dtype=dtype, name=name)
        return input_layer

    def get_input_double_layer(self, name=None, dtype="float32"):
        '''
        input层 dense seqs
        '''
        input_layer = keras.Input(
            shape=(seq_length_creative_id,), dtype=dtype, name=name)
        return input_layer

    def gru_net(self, emb_layer, click_times_weight):
        emb_layer = keras.layers.SpatialDropout1D(0.3)(emb_layer)
        x = keras.layers.Conv1D(
            filters=emb_layer.shape[-1], kernel_size=1, padding='same', activation='relu')(emb_layer)

        # 以上为embedding部分
        # bilstm
        x = keras.layers.Bidirectional(keras.layers.LSTM(
            self.n_units, dropout=0.2, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(
            self.n_units, dropout=0.2, return_sequences=True))(x)
        conv1a = keras.layers.Conv1D(filters=128, kernel_size=2,
                                     padding='same', activation='relu',)(x)
        conv1b = keras.layers.Conv1D(filters=64, kernel_size=4,
                                     padding='same', activation='relu', )(x)
        conv1c = keras.layers.Conv1D(filters=32, kernel_size=8,
                                     padding='same', activation='relu',)(x)
        gap1a = keras.layers.GlobalAveragePooling1D()(conv1a)
        gap1b = keras.layers.GlobalAveragePooling1D()(conv1b)
        gap1c = keras.layers.GlobalMaxPooling1D()(conv1c)
        max_pool1 = keras.layers.GlobalMaxPooling1D()(x)
        concat = keras.layers.concatenate([max_pool1, gap1a, gap1b, gap1c])
        return concat

    def get_embedding_conv1ded(self, embedding_vector, filter_size=128):
        x = keras.layers.Conv1D(filters=filter_size, kernel_size=1,
                                padding='same', activation='relu')(embedding_vector)
        return x

    def create_model(self, num_class, labeli):
        """
        构建模型的函数
        """
        K.clear_session()
        # cols to use
        inputlist = cols_to_emb
        # 这个字典用于指定哪些embedding层也可以进行训练
        train_able_dict = {'creative_id': False, 'ad_id': False, 'advertiser_id': False,
                           'product_id': False, 'industry': True, 'product_category': True, 'time': True, 'click_times': True}
        # 所有的input层
        inputs_all = []
        for col in inputlist:
            inputs_all.append(self.get_input_layer(name=col))
        # inputs_all.append(self.get_input_double_layer(name = 'click_times'))# 没用上

        # input->seq embedding
        emb_layer_concat_dict = {}
        for index, col in enumerate(inputlist):
            layer_emb = self.get_emb_layer(
                emb_matrix_dict[col][0], input_length=seq_length_creative_id, trainable=train_able_dict[col])(inputs_all[index])
            emb_layer_concat_dict[col] = layer_emb

        # 每个列各自降维提取信息
        for col in inputlist:
            if conv1d_info_dict[col] > 0:
                emb_layer_concat_dict[col] = self.get_embedding_conv1ded(
                    emb_layer_concat_dict[col], filter_size=conv1d_info_dict[col])

        # 所有列拼接到一起
        concat_all = keras.layers.concatenate(
            list(emb_layer_concat_dict.values()))
        # 进bilstm
        concat_all = self.gru_net(concat_all, inputs_all[-1])

        concat_all = keras.layers.Dropout(0.3)(concat_all)
        x = keras.layers.Dense(256)(concat_all)
        x = keras.layers.PReLU()(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.PReLU()(x)

        outputs_all = keras.layers.Dense(
            num_class, activation='softmax', name=labeli)(x)  # 10分类
        model = keras.Model(inputs_all, outputs_all)
        print(model.summary())
        optimizer = keras.optimizers.Adam(1e-3)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00001,
    patience=3,
    verbose=1,
    mode="max",
    baseline=None,
    restore_best_weights=True,
)

csv_log_callback = tf.keras.callbacks.CSVLogger(
    filename='logs_save/{}_nn_v0621_{}d_bilstm.log'.format(labeli, count), separator=",", append=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                          factor=0.5,
                                                          patience=1,
                                                          min_lr=0.0000001)

callbacks = [earlystop_callback, csv_log_callback, reduce_lr_callback]
