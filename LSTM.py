# %%
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, Concatenate, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec, KeyedVectors
from layers import Add, LayerNormalization
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import PositionEncoding
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

# %%


def get_data():
    DATA = {}
    DATA['X1_train'] = np.load('tmp/inputs_0.npy', allow_pickle=True)
    DATA['X1_val'] = np.load('tmp/inputs_1.npy', allow_pickle=True)
    DATA['X2_train'] = np.load('tmp/inputs_2.npy', allow_pickle=True)
    DATA['X2_val'] = np.load('tmp/inputs_3.npy', allow_pickle=True)
    DATA['X3_train'] = np.load('tmp/inputs_4.npy', allow_pickle=True)
    DATA['X3_val'] = np.load('tmp/inputs_5.npy', allow_pickle=True)
    DATA['X4_train'] = np.load('tmp/inputs_6.npy', allow_pickle=True)
    DATA['X4_val'] = np.load('tmp/inputs_7.npy', allow_pickle=True)
    DATA['X5_train'] = np.load('tmp/inputs_8.npy', allow_pickle=True)
    DATA['X5_val'] = np.load('tmp/inputs_9.npy', allow_pickle=True)
    DATA['X6_train'] = np.load('tmp/inputs_10.npy', allow_pickle=True)
    DATA['X6_val'] = np.load('tmp/inputs_11.npy', allow_pickle=True)
    DATA['Y_gender_train'] = np.load('tmp/gender_0.npy', allow_pickle=True)
    DATA['Y_gender_val'] = np.load('tmp/gender_1.npy', allow_pickle=True)
    DATA['Y_age_train'] = np.load('tmp/age_0.npy', allow_pickle=True)
    DATA['Y_age_val'] = np.load('tmp/age_1.npy', allow_pickle=True)
    DATA['creative_id_emb'] = np.load(
        'tmp/embeddings_0.npy', allow_pickle=True)
    DATA['ad_id_emb'] = np.load(
        'tmp/embeddings_1.npy', allow_pickle=True)
    DATA['product_id_emb'] = np.load(
        'tmp/embeddings_2.npy', allow_pickle=True)
    DATA['advertiser_id_emb'] = np.load(
        'tmp/embeddings_3.npy', allow_pickle=True)
    DATA['industry_emb'] = np.load(
        'tmp/embeddings_4.npy', allow_pickle=True)
    DATA['product_category_emb'] = np.load(
        'tmp/embeddings_5.npy', allow_pickle=True)

    # DATA['Y_age_train'] = pd.read_csv(
    #     'data/train_preliminary/user.csv').age.values-1
    # DATA['Y_age_val'] = pd.read_csv(
    #     'data/train_preliminary/user.csv').age.values-1
    # DATA['Y_gender_train'] = pd.read_csv(
    #     'data/train_preliminary/user.csv').gender.values-1
    # DATA['Y_gender_val'] = pd.read_csv(
    #     'data/train_preliminary/user.csv').gender.values-1

    return DATA


# %%
DATA = get_data()

cols_to_emb = ['creative_id', 'ad_id', 'advertiser_id',
               'product_id', 'industry', 'product_category']

emb_matrix_dict = {
    'creative_id': [DATA['creative_id_emb'].astype('float32')],
    'ad_id': [DATA['ad_id_emb'].astype('float32')],
    'product_id': [DATA['product_id_emb'].astype('float32')],
    'advertiser_id': [DATA['advertiser_id_emb'].astype('float32')],
    'industry': [DATA['industry_emb'].astype('float32')],
    'product_category': [DATA['product_category_emb'].astype('float32')],
}

conv1d_info_dict = {'creative_id': 128, 'ad_id': 128, 'advertiser_id': 128,
                    'industry': 128, 'product_category': 128,
                    'product_id': 128, 'time': 32, 'click_times': -1}
# %%
seq_length_creative_id = 100
labeli = 'age'
# %%


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

    def get_input_layer(self, name=None, dtype="int64"):
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
                      #   loss='sparse_categorical_crossentropy',
                      loss=tf.keras.losses.CategoricalCrossentropy(
                          from_logits=False),
                      metrics=['accuracy'])
        return model


# %%
model = BiLSTM_Model(n_units=128).create_model(10, 'age')

# %%
# train_examples = 720000
# val_examples = 180000
train_examples = 810000
val_examples = 90000
model.fit(
    {
        'creative_id': DATA['X1_train'][:train_examples],
        'ad_id': DATA['X2_train'][:train_examples],
        'product_id': DATA['X3_train'][:train_examples],
        'advertiser_id': DATA['X4_train'][:train_examples],
        'industry': DATA['X5_train'][:train_examples],
        'product_category': DATA['X6_train'][:train_examples]
    },
    {
        # 'gender': DATA['Y_gender_train'][:train_examples],
        'age': DATA['Y_age_train'][:train_examples],
    },
    validation_data=(
        {
            'creative_id': DATA['X1_val'][:val_examples],
            'ad_id': DATA['X2_val'][:val_examples],
            'product_id': DATA['X3_val'][:val_examples],
            'advertiser_id': DATA['X4_val'][:val_examples],
            'industry': DATA['X5_val'][:val_examples],
            'product_category': DATA['X6_val'][:val_examples]
        },
        {
            # 'gender': DATA['Y_gender_val'][:val_examples],
            'age': DATA['Y_age_val'][:val_examples],
        },
    ),
    epochs=10,
    batch_size=1024,
    # callbacks=[checkpoint, earlystop_callback, reduce_lr_callback],
)
# %%
# earlystop_callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     min_delta=0.00001,
#     patience=3,
#     verbose=1,
#     mode="max",
#     baseline=None,
#     restore_best_weights=True,
# )

# csv_log_callback = tf.keras.callbacks.CSVLogger(
#     filename='logs_save/{}_nn_v0621_{}d_bilstm.log'.format(labeli, count), separator=",", append=True)

# reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
#                                                           factor=0.5,
#                                                           patience=1,
#                                                           min_lr=0.0000001)

# callbacks = [earlystop_callback, csv_log_callback, reduce_lr_callback]
