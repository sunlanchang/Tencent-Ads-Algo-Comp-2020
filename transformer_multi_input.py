# %%
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, concatenate, Bidirectional
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from mymail import mail
import os
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


# %%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# %%


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
# f = open('word2vec/userid_creative_ids.txt')
# LEN_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     LEN_creative_id = max(LEN_creative_id, current_line_len)
# f.close()


# %%
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


# %%
NUM_creative_id = 2481135+1
NUM_ad_id = 2264190+1
NUM_product_id = 33273+1

LEN_creative_id = 100
LEN_ad_id = 100
LEN_product_id = 100

vocab_size = NUM_creative_id 
maxlen = LEN_creative_id


def get_age_model(creative_id_emb, ad_id_emb, product_id_emb):
    embed_dim = 128  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    # x1 = Embedding(input_dim=NUM_creative_id,
    #                output_dim=128,
    #                weights=[creative_id_emb],
    #                trainable=True,
    #                input_length=LEN_creative_id,
    #                mask_zero=True)(input_creative_id)
    x1 = TokenAndPositionEmbedding(
        maxlen, vocab_size, embed_dim)(input_creative_id)
    x1 = TransformerBlock(embed_dim, num_heads, ff_dim)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    x1 = layers.Dropout(0.1)(x1)
    x1 = layers.Dense(20, activation="relu")(x1)
    x1 = layers.Dropout(0.1)(x1)
    outputs = layers.Dense(10, activation="softmax")(x1)
    # x1 = LSTM(1024, return_sequences=True)(x1)
    # x1 = LSTM(512, return_sequences=True)(x1)
    # x1 = LSTM(256, return_sequences=False)(x1)

    # second input
    # input_ad_id = Input(shape=(None,), name='ad_id')
    # x2 = Embedding(input_dim=NUM_ad_id,
    #                output_dim=128,
    #                weights=[ad_id_emb],
    #                trainable=True,
    #                input_length=LEN_ad_id,
    #                mask_zero=True)(input_ad_id)
    # x2 = LSTM(1024, return_sequences=True)(x2)
    # x2 = LSTM(512, return_sequences=True)(x2)
    # x2 = LSTM(256, return_sequences=False)(x2)

    # third input
    # input_product_id = Input(shape=(None,), name='product_id')
    # x3 = Embedding(input_dim=NUM_product_id,
    #                output_dim=128,
    #                weights=[product_id_emb],
    #                trainable=True,
    #                input_length=LEN_product_id,
    #                mask_zero=True)(input_product_id)
    # x3 = LSTM(1024, return_sequences=True)(x3)
    # x3 = LSTM(512, return_sequences=True)(x3)
    # x3 = LSTM(256, return_sequences=False)(x3)

    # concat x1 x2
    # x = concatenate([x1, x2, x3])
    # x = x1 + x2 + x3
    # x = Dense(128)(x)
    # x = Dropout(0.1)(x)
    # output_y = Dense(10, activation='softmax')(x)

    # model = Model([input_creative_id, input_ad_id, input_product_id], output_y)
    model = Model(input_creative_id, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# %%
mail('start getting train data')
x1_train, x1_val, x2_train, x2_val, x3_train, x3_val, y_train, y_val, creative_id_emb, ad_id_emb, product_id_emb = get_train_val()
mail('get train data done.')

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
    mail('start train lstm')
    model.fit(
        {'creative_id': x1_train, 'ad_id': x2_train, 'product_id': x3_train},
        y_train,
        validation_data=([x1_val, x2_val, x3_val], y_val),
        epochs=3,
        batch_size=256,
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
