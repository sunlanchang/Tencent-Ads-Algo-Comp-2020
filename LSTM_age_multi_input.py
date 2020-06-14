# %%
# 生成词嵌入文件
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Embedding, Dense, Dropout, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec, KeyedVectors
from mymail import mail
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# f = open('word2vec/userid_creative_ids.txt')
# LEN_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     LEN_creative_id = max(LEN_creative_id, current_line_len)
# f.close()
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--load_from_npy', action='store_true',
                    help='从npy文件加载数据',
                    default=False)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',
                    default=True)

parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集',
                    default=810000)


parser.add_argument('--num_lstm', type=int,
                    help='LSTM head个数',
                    default=1)

args = parser.parse_args()
# %%
NUM_creative_id = 2481135+1
NUM_ad_id = 2264190+1
NUM_product_id = 33273+1


def get_train_val():

    # 获取 creative_id 特征
    # f = open('tmp/userid_creative_ids.txt')
    f = open('word2vec/userid_creative_ids.txt')
    tokenizer = Tokenizer(num_words=NUM_creative_id)
    tokenizer.fit_on_texts(f)
    f.close()
    creative_id_seq = []
    with open('word2vec/userid_creative_ids.txt') as f:
        for text in f:
            creative_id_seq.append(text.strip())

    sequences = tokenizer.texts_to_sequences(creative_id_seq[:900000//1])
    X1_train = pad_sequences(
        sequences, maxlen=LEN_creative_id, padding='post')

    # 获取creative_id embedding
    def get_creative_id_emb():
        path = "word2vec/wordvectors_creative_id.kv"
        wv = KeyedVectors.load(path, mmap='r')
        creative_id_tokens = list(wv.vocab.keys())
        embedding_dim = 128
        embedding_matrix = np.random.randn(
            len(creative_id_tokens)+1, embedding_dim)
        for creative_id in creative_id_tokens:
            embedding_vector = wv[creative_id]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([creative_id])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    creative_id_emb = get_creative_id_emb()

    # 获取 ad_id 特征
    f = open('word2vec/userid_ad_ids.txt')
    tokenizer = Tokenizer(num_words=NUM_ad_id)
    tokenizer.fit_on_texts(f)
    f.close()
    ad_id_seq = []
    with open('word2vec/userid_ad_ids.txt') as f:
        for text in f:
            ad_id_seq.append(text.strip())

    sequences = tokenizer.texts_to_sequences(ad_id_seq[:900000//1])
    X2_train = pad_sequences(
        sequences, maxlen=LEN_ad_id, padding='post')

    def get_ad_id_emb():
        path = "word2vec/wordvectors_ad_id.kv"
        wv = KeyedVectors.load(path, mmap='r')
        ad_id_tokens = list(wv.vocab.keys())
        embedding_dim = 128
        embedding_matrix = np.random.randn(
            len(ad_id_tokens)+1, embedding_dim)
        for ad_id in ad_id_tokens:
            embedding_vector = wv[ad_id]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([ad_id])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    ad_id_emb = get_ad_id_emb()

    # 获取 product_id 特征
    # f = open('tmp/userid_product_ids.txt')
    f = open('word2vec/userid_product_ids.txt')
    tokenizer = Tokenizer(num_words=NUM_product_id)
    tokenizer.fit_on_texts(f)
    f.close()
    product_id_seq = []
    with open('word2vec/userid_product_ids.txt') as f:
        for text in f:
            product_id_seq.append(text.strip())

    sequences = tokenizer.texts_to_sequences(product_id_seq[:900000//1])
    X3_train = pad_sequences(
        sequences, maxlen=LEN_product_id, padding='post')

    # 获取product_id embedding
    def get_product_id_emb():
        path = "word2vec/wordvectors_product_id.kv"
        wv = KeyedVectors.load(path, mmap='r')
        product_id_tokens = list(wv.vocab.keys())
        embedding_dim = 128
        embedding_matrix = np.random.randn(
            len(product_id_tokens)+1, embedding_dim)
        for product_id in product_id_tokens:
            embedding_vector = wv[product_id]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([product_id])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    product_id_emb = get_product_id_emb()

    # 获得age标签
    user_train = pd.read_csv(
        'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
    Y_gender = user_train['gender'].values
    Y_age = user_train['age'].values
    Y_gender = Y_gender - 1
    Y_age = Y_age - 1
    Y_age = to_categorical(Y_age)
    num_examples = Y_age.shape[0]
    train_examples = int(num_examples * 0.9)

    # 分别对应 x1_train x1_val x2_train x2_val y_train y_val
    return X1_train[:train_examples], X1_train[train_examples:], X2_train[:train_examples], X2_train[train_examples:], X3_train[:train_examples], X3_train[train_examples:], Y_age[:train_examples], Y_age[train_examples:], creative_id_emb, ad_id_emb, product_id_emb

# %%


def get_test():
    pass


# %%
LEN_creative_id = 100
LEN_ad_id = 100
LEN_product_id = 100


def get_age_model(creative_id_emb, ad_id_emb, product_id_emb):
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id,
                   output_dim=128,
                   weights=[creative_id_emb],
                   trainable=args.not_train_embedding,
                   input_length=LEN_creative_id,
                   mask_zero=True)(input_creative_id)
    for _ in range(args.num_lstm):
        x1 = Bidirectional(LSTM(256, return_sequences=True))(x1)
    x1 = layers.GlobalMaxPooling1D()(x1)

    # second input
    input_ad_id = Input(shape=(None,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id,
                   output_dim=128,
                   weights=[ad_id_emb],
                   trainable=args.not_train_embedding,
                   input_length=LEN_ad_id,
                   mask_zero=True)(input_ad_id)
    for _ in range(args.num_lstm):
        x2 = Bidirectional(LSTM(256, return_sequences=True))(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # third input
    input_product_id = Input(shape=(None,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id,
                   output_dim=128,
                   weights=[product_id_emb],
                   trainable=args.not_train_embedding,
                   input_length=LEN_product_id,
                   mask_zero=True)(input_product_id)
    for _ in range(args.num_lstm):
        x3 = Bidirectional(LSTM(256, return_sequences=True))(x3)
    x3 = layers.GlobalMaxPooling1D()(x3)

    # concat x1 x2
    x = concatenate([x1, x2, x3])
    # x = Dense(128)(x)
    # x = Dropout(0.1)(x)
    output_y = Dense(10, activation='softmax')(x)

    model = Model([input_creative_id, input_ad_id, input_product_id], output_y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# %%
if not args.load_from_npy:
    mail('start getting train data')
    x1_train, x1_val, x2_train, x2_val, x3_train, x3_val, y_train, y_val, creative_id_emb, ad_id_emb, product_id_emb = get_train_val()
    mail('get train data done.')

    def save_data(datas):
        for i, data in enumerate(datas):
            np.save(f'tmp/transformer_input_{i}.npy', data)
    datas = [x1_train, x1_val, x2_train, x2_val, x3_train, x3_val,
             y_train, y_val, creative_id_emb, ad_id_emb, product_id_emb]
    save_data(datas)
else:
    x1_train = np.load('tmp/transformer_input_0.npy', allow_pickle=True)
    x1_val = np.load('tmp/transformer_input_1.npy', allow_pickle=True)
    x2_train = np.load('tmp/transformer_input_2.npy', allow_pickle=True)
    x2_val = np.load('tmp/transformer_input_3.npy', allow_pickle=True)
    x3_train = np.load('tmp/transformer_input_4.npy', allow_pickle=True)
    x3_val = np.load('tmp/transformer_input_5.npy', allow_pickle=True)
    y_train = np.load('tmp/transformer_input_6.npy', allow_pickle=True)
    y_val = np.load('tmp/transformer_input_7.npy', allow_pickle=True)
    creative_id_emb = np.load('tmp/transformer_input_8.npy', allow_pickle=True)
    ad_id_emb = np.load('tmp/transformer_input_9.npy', allow_pickle=True)
    product_id_emb = np.load('tmp/transformer_input_10.npy', allow_pickle=True)

# %%
model = get_age_model(creative_id_emb, ad_id_emb, product_id_emb)
# %%
# %%
# 测试数据格式(batch_size, sequence长度)
# x1 = np.array([1, 2, 3, 4]).reshape(1, -1)
# x2 = np.array([1, 2, 3, 4]).reshape(1, -1)
# model.predict([x1, x2])


# %%
checkpoint = ModelCheckpoint("tmp/age_epoch_{epoch:02d}.hdf5", monitor='val_loss', verbose=1,
                             save_best_only=False, mode='auto', period=1)
# %%
# model.fit(
#     {'creative_id': x1_train, 'ad_id': x2_train},
#     y_train,
#     validation_data=([x1_val, x2_val], y_val),
#     epochs=5,
#     batch_size=256,
#     callbacks=[checkpoint],
# )

# %%
try:
    examples = args.examples
    mail('start train lstm')
    model.fit(
        {'creative_id': x1_train[:examples], 'ad_id': x2_train[:examples],
            'product_id': x3_train[:examples]},
        y_train[:examples],
        validation_data=([x1_val, x2_val, x3_val], y_val),
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
    )
    mail('train lstm done!!!')
except Exception as e:
    e = str(e)
    mail('train lstm failed!!! ' + e)


# %%
# model.load_weights('tmp\gender_epoch_01.hdf5')


# # %%
# if debug:
#     sequences = tokenizer.texts_to_sequences(
#         creative_id_seq[900000:])
# else:
#     sequences = tokenizer.texts_to_sequences(
#         creative_id_seq[900000:])

# X_test = pad_sequences(sequences, maxlen=LEN_creative_id)
# # %%
# y_pred = model.predict(X_test, batch_size=4096)

# y_pred = np.where(y_pred > 0.5, 1, 0)
# y_pred = y_pred.flatten()

# # %%
# y_pred = y_pred+1
# # %%
# res = pd.DataFrame({'predicted_gender': y_pred})
# res.to_csv(
#     'data/ans/lstm_gender.csv', header=True, columns=['predicted_gender'], index=False)


# # %%
# mail('predict lstm gender done')

# %%
