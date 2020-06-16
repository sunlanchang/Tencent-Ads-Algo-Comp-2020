# %%
# 生成词嵌入文件
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
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, Concatenate, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec, KeyedVectors
from mymail import mail
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
python Transformer_keras.py --load_from_npy --batch_size 256 --epoch 5 --num_transformer 1 --head_attention 1 --num_lstm 1 --examples 100000
'''

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--load_from_npy', action='store_true',
                    help='从npy文件加载数据',
                    default=False)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',
                    default=True)
parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--head_attention', type=int,
                    help='transformer head个数',
                    default=1)
parser.add_argument('--num_lstm', type=int,
                    help='LSTM 个数',
                    default=1)
parser.add_argument('--train_examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)
parser.add_argument('--val_examples', type=int,
                    help='验证集数据，调试时候可以设置1000',
                    default=90000)
args = parser.parse_args()
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'projection_dim': self.projection_dim,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'combine_heads': self.combine_heads,
        })
        return config

# %%


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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config
# %%


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim, pre_training_embedding):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=emded_dim,
            weights=[pre_training_embedding],
            input_length=100,
            trainable=args.not_train_embedding,
            mask_zero=True,
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        # 忽略位置编码
        # return x + positions
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config
# f = open('word2vec/userid_creative_ids.txt')
# LEN_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     LEN_creative_id = max(LEN_creative_id, current_line_len)
# f.close()


# %%
NUM_creative_id = 2481135  # embedding词表大小+1，其中+1为了未出现在此表中的UNK词
NUM_ad_id = 2264190
NUM_product_id = 33273
NUM_advertiser_id = 52090
NUM_industry = 326
NUM_product_category = 18

LEN_creative_id = 100
LEN_ad_id = 100
LEN_product_id = 100
LEN_advertiser_id = 100
LEN_industry = 100
LEN_product_category = 100

# vocab_size = NUM_creative_id
maxlen = 100


def get_model(creative_id_emb, ad_id_emb, product_id_emb):
    embed_dim = 128  # Embedding size for each token
    num_heads = 1  # Number of attention heads
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = TokenAndPositionEmbedding(
        maxlen, NUM_creative_id, embed_dim, creative_id_emb)(input_creative_id)
    for _ in range(args.num_transformer):
        x1 = TransformerBlock(embed_dim, num_heads, ff_dim)(x1)

    for _ in range(args.num_lstm):
        x1 = Bidirectional(LSTM(256, return_sequences=True))(x1)
    x1 = layers.GlobalMaxPooling1D()(x1)

    # second input
    input_ad_id = Input(shape=(None,), name='ad_id')

    x2 = TokenAndPositionEmbedding(
        maxlen, NUM_ad_id, embed_dim, ad_id_emb)(input_ad_id)
    for _ in range(args.num_transformer):
        x2 = TransformerBlock(embed_dim, num_heads, ff_dim)(x2)
    for _ in range(args.num_lstm):
        x2 = Bidirectional(LSTM(256, return_sequences=True))(x2)
    # x2 = Bidirectional(LSTM(256, return_sequences=False))(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # third input
    input_product_id = Input(shape=(None,), name='product_id')

    x3 = TokenAndPositionEmbedding(
        maxlen, NUM_product_id, embed_dim, product_id_emb)(input_product_id)
    for _ in range(args.num_transformer):
        x3 = TransformerBlock(embed_dim, num_heads, ff_dim)(x3)
    for _ in range(args.num_lstm):
        x3 = Bidirectional(LSTM(256, return_sequences=True))(x3)
    # x3 = Bidirectional(LSTM(256, return_sequences=False))(x3)
    x3 = layers.GlobalMaxPooling1D()(x3)

    # concat x1 x2 x3
    x = Concatenate(axis=1)([x1, x2, x3])
    # x = x1 + x2 + x3
    x = Dense(20)(x)
    # x = Dropout(0.1)(x)
    output_y = Dense(10, activation='softmax')(x)

    model = Model([input_creative_id, input_ad_id, input_product_id], output_y)
    # model = Model(input_creative_id, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# %%
def get_model_head_concat(DATA):
    embed_dim = 128  # Embedding size for each token
    num_heads = args.head_attention  # Number of attention heads
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = TokenAndPositionEmbedding(
        maxlen, NUM_creative_id+1, embed_dim, DATA['creative_id_emb'])(input_creative_id)

    input_ad_id = Input(shape=(None,), name='ad_id')
    x2 = TokenAndPositionEmbedding(
        maxlen, NUM_ad_id+1, embed_dim, DATA['ad_id_emb'])(input_ad_id)

    input_product_id = Input(shape=(None,), name='product_id')
    x3 = TokenAndPositionEmbedding(
        maxlen, NUM_product_id+1, embed_dim, DATA['product_id_emb'])(input_product_id)

    input_advertiser_id = Input(shape=(None,), name='advertiser_id')
    x4 = TokenAndPositionEmbedding(
        maxlen, NUM_advertiser_id+1, embed_dim, DATA['advertiser_id_emb'])(input_advertiser_id)

    input_industry = Input(shape=(None,), name='industry')
    x5 = TokenAndPositionEmbedding(
        maxlen, NUM_industry+1, embed_dim, DATA['industry_emb'])(input_industry)

    input_product_category = Input(shape=(None,), name='product_category')
    x6 = TokenAndPositionEmbedding(
        maxlen, NUM_product_category+1, embed_dim, DATA['product_category_emb'])(input_product_category)

    # concat
    # x = x1 + x2 + x3
    x = layers.Concatenate(axis=1)([x1, x2, x3, x4, x5, x6])

    for _ in range(args.num_transformer):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    for _ in range(args.num_lstm):
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)

    output_gender = Dense(2, activation='softmax', name='gender')(x)
    output_age = Dense(10, activation='softmax', name='age')(x)

    model = Model(
        [
            input_creative_id,
            input_ad_id,
            input_product_id,
            input_advertiser_id,
            input_industry,
            input_product_category
        ],
        [
            output_gender,
            output_age
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss={'gender': losses.CategoricalCrossentropy(from_logits=False),
              'age': losses.CategoricalCrossentropy(from_logits=False)},
        loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    model.summary()

    return model


# %%


def get_train_val():

    # 提取词向量文件
    def get_embedding(feature_name, tokenizer):
        path = f"word2vec/wordvectors_{feature_name}.kv"
        wv = KeyedVectors.load(path, mmap='r')
        feature_tokens = list(wv.vocab.keys())
        embedding_dim = 128
        embedding_matrix = np.random.randn(
            len(feature_tokens)+1, embedding_dim)
        for feature in feature_tokens:
            embedding_vector = wv[feature]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([feature])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    # 从序列文件提取array格式数据
    def get_train(feature_name, vocab_size, len_feature):
        f = open(f'word2vec/userid_{feature_name}s.txt')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(f)
        f.close()

        feature_seq = []
        with open(f'word2vec/userid_{feature_name}s.txt') as f:
            for text in f:
                feature_seq.append(text.strip())

        sequences = tokenizer.texts_to_sequences(feature_seq[:900000//1])
        X_train = pad_sequences(
            sequences, maxlen=len_feature, padding='post')
        return X_train, tokenizer

    # 构造输出的训练标签
    # 获得age、gender标签
    DATA = {}

    user_train = pd.read_csv(
        'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
    Y_gender = user_train['gender'].values
    Y_age = user_train['age'].values
    Y_gender = Y_gender - 1
    Y_age = Y_age - 1
    Y_age = to_categorical(Y_age)
    Y_gender = to_categorical(Y_gender)

    num_examples = Y_age.shape[0]
    train_examples = int(num_examples * 0.9)

    DATA['Y_gender_train'] = Y_gender[:train_examples]
    DATA['Y_gender_val'] = Y_gender[train_examples:]
    DATA['Y_age_train'] = Y_age[:train_examples]
    DATA['Y_age_val'] = Y_age[train_examples:]

    # 第一个输入
    print('获取 creative_id 特征')
    X1_train, tokenizer = get_train(
        'creative_id', NUM_creative_id+1, LEN_creative_id)  # +1为了UNK的creative_id
    creative_id_emb = get_embedding('creative_id', tokenizer)

    DATA['X1_train'] = X1_train[:train_examples]
    DATA['X1_val'] = X1_train[train_examples:]
    DATA['creative_id_emb'] = creative_id_emb

    # 第二个输入
    print('获取 ad_id 特征')
    X2_train, tokenizer = get_train(
        'ad_id', NUM_ad_id+1, LEN_ad_id)
    ad_id_emb = get_embedding('ad_id', tokenizer)

    DATA['X2_train'] = X2_train[:train_examples]
    DATA['X2_val'] = X2_train[train_examples:]
    DATA['ad_id_emb'] = ad_id_emb

    # 第三个输入
    print('获取 product_id 特征')
    X3_train, tokenizer = get_train(
        'product_id', NUM_product_id+1, LEN_product_id)
    product_id_emb = get_embedding('product_id', tokenizer)

    DATA['X3_train'] = X3_train[:train_examples]
    DATA['X3_val'] = X3_train[train_examples:]
    DATA['product_id_emb'] = product_id_emb

    # 第四个输入
    print('获取 advertiser_id 特征')
    X4_train, tokenizer = get_train(
        'advertiser_id', NUM_advertiser_id+1, LEN_advertiser_id)
    advertiser_id_emb = get_embedding('advertiser_id', tokenizer)

    DATA['X4_train'] = X4_train[:train_examples]
    DATA['X4_val'] = X4_train[train_examples:]
    DATA['advertiser_id_emb'] = advertiser_id_emb

    # 第五个输入
    print('获取 industry 特征')
    X5_train, tokenizer = get_train(
        'industry', NUM_industry+1, LEN_industry)
    industry_emb = get_embedding('industry', tokenizer)

    DATA['X5_train'] = X5_train[:train_examples]
    DATA['X5_val'] = X5_train[train_examples:]
    DATA['industry_emb'] = industry_emb

    # 第六个输入
    print('获取 product_category 特征')
    X6_train, tokenizer = get_train(
        'product_category', NUM_product_category+1, LEN_product_category)
    product_category_emb = get_embedding('product_category', tokenizer)

    DATA['X6_train'] = X6_train[:train_examples]
    DATA['X6_val'] = X6_train[train_examples:]
    DATA['product_category_emb'] = product_category_emb

    return DATA


# %%
if not args.load_from_npy:
    mail('start getting train data')
    print('从csv文件提取训练数据到array格式，大概十几分钟时间')
    DATA = get_train_val()
    mail('get train data done.')

    # 训练数据保存为npy文件
    dirs = 'tmp/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    def save_npy(datas, name):
        for i, data in enumerate(datas):
            np.save(f'tmp/{name}_{i}.npy', data)
            print(f'saving tmp/{name}_{i}.npy')

    inputs = [
        DATA['X1_train'], DATA['X1_val'],
        DATA['X2_train'], DATA['X2_val'],
        DATA['X3_train'], DATA['X3_val'],
        DATA['X4_train'], DATA['X4_val'],
        DATA['X5_train'], DATA['X5_val'],
        DATA['X6_train'], DATA['X6_val'],
    ]
    outputs_gender = [DATA['Y_gender_train'], DATA['Y_gender_val']]
    outputs_age = [DATA['Y_age_train'], DATA['Y_age_val']]
    embeddings = [
        DATA['creative_id_emb'],
        DATA['ad_id_emb'],
        DATA['product_id_emb'],
        DATA['advertiser_id_emb'],
        DATA['industry_emb'],
        DATA['product_category_emb'],
    ]
    save_npy(inputs, 'inputs')
    save_npy(outputs_gender, 'gender')
    save_npy(outputs_age, 'age')
    save_npy(embeddings, 'embeddings')
else:
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


# %%
# model = get_age_model(creative_id_emb, ad_id_emb, product_id_emb)
model = get_model_head_concat(DATA)
# %%
# 测试数据格式(batch_size, sequence长度)
# x1 = np.array([1, 2, 3, 4]).reshape(1, -1)
# x2 = np.array([1, 2, 3, 4]).reshape(1, -1)
# model.predict([x1, x2])


# %%
checkpoint = ModelCheckpoint("tmp/transformer_epoch_{epoch:02d}.hdf5", save_weights_only=True, monitor='val_loss', verbose=1,
                             save_best_only=False, mode='auto', period=1)
# %%
try:
    train_examples = args.train_examples
    val_examples = args.val_examples
    mail('start train')
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
            'gender': DATA['Y_gender_train'][:train_examples],
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
                'gender': DATA['Y_gender_val'][:val_examples],
                'age': DATA['Y_age_val'][:val_examples],
            },
        ),
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
    )
    mail('train done!!!')
except Exception as e:
    e = str(e)
    mail('train failed!!! ' + e)

# %%
# model.load_weights('tmp/gender_epoch_01.hdf5')


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
