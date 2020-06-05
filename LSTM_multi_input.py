# %%
# 生成词嵌入文件
from tqdm import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, concatenate
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from mymail import mail
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%

debug = True

# %%


# %%
# f = open('word2vec/userid_creative_ids.txt')
# LEN_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     LEN_creative_id = max(LEN_creative_id, current_line_len)
# f.close()


# %%


# %%
NUM_creative_id = 2481135+1
NUM_ad_id = 2264190+1


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

    # 获得gender标签
    user_train = pd.read_csv(
        'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
    Y_gender = user_train['gender'].values
    Y_age = user_train['age'].values
    Y_gender = Y_gender - 1

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

    num_examples = Y_gender.shape[0]
    train_examples = int(num_examples * 0.9)

    # 分别对应 x1_train x1_val x2_train x2_val y_train y_val
    return X1_train[:train_examples], X1_train[train_examples:], X2_train[:train_examples], X2_train[train_examples:], Y_gender[:train_examples], Y_gender[train_examples:], creative_id_emb, ad_id_emb

# %%


def get_test():
    pass


def get_embedding():
    pass


# %%
LEN_creative_id = 100
LEN_ad_id = 100


def get_gender_model(creative_id_emb, ad_id_emb):
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id,
                   output_dim=128,
                   weights=[creative_id_emb],
                   trainable=True,
                   input_length=LEN_creative_id,
                   mask_zero=True)(input_creative_id)
    x1 = LSTM(1024, return_sequences=True)(x1)
    x1 = LSTM(512, return_sequences=False)(x1)

    # second input
    input_ad_id = Input(shape=(None,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id,
                   output_dim=128,
                   weights=[ad_id_emb],
                   trainable=True,
                   input_length=LEN_ad_id,
                   mask_zero=True)(input_ad_id)
    x2 = LSTM(1024, return_sequences=True)(x2)
    x2 = LSTM(512, return_sequences=False)(x2)

    # concat x1 x2
    x = concatenate([x1, x2])
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    output_y = Dense(1, activation='sigmoid')(x)

    model = Model([input_creative_id, input_ad_id], output_y)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# %%
x1_train, x1_val, x2_train, x2_val, y_train, y_val, creative_id_emb, ad_id_emb = get_train_val()


# %%
model = get_gender_model(creative_id_emb, ad_id_emb)
# %%
# 测试数据格式(batch_size, sequence长度)
x1 = np.array([1, 2, 3, 4]).reshape(1, -1)
x2 = np.array([1, 2, 3, 4]).reshape(1, -1)
model.predict([x1, x2])


# %%
# %%
checkpoint = ModelCheckpoint("tmp/gender_epoch_{epoch:02d}.hdf5", monitor='val_loss', verbose=0,
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
    mail('start train lstm')
    model.fit(
        {'creative_id': x1_train, 'ad_id': x2_train},
        y_train,
        validation_data=([x1_val, x2_val], y_val),
        epochs=5,
        batch_size=256,
        callbacks=[checkpoint],
    )
    mail('train gender lstm done!!!')
except Exception as e:
    e = str(e)
    mail('train lstm failed!!! ' + e)


# %%
model.load_weights('tmp\gender_epoch_01.hdf5')


# %%
if debug:
    sequences = tokenizer.texts_to_sequences(
        creative_id_seq[900000:])
else:
    sequences = tokenizer.texts_to_sequences(
        creative_id_seq[900000:])

X_test = pad_sequences(sequences, maxlen=LEN_creative_id)
# %%
y_pred = model.predict(X_test, batch_size=4096)

y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = y_pred.flatten()

# %%
y_pred = y_pred+1
# %%
res = pd.DataFrame({'predicted_gender': y_pred})
res.to_csv(
    'data/ans/lstm_gender.csv', header=True, columns=['predicted_gender'], index=False)


# %%
mail('predict lstm gender done')

# %%
