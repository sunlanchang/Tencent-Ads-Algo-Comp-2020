# %%
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
# 统计creative_id序列的长度，只需要统计一次
# f = open('word2vec/userid_creative_ids.txt')
# LEN_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     LEN_creative_id = max(LEN_creative_id, current_line_len)
# f.close()
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--load_from_npy', action='store_true',
                    help='从npy文件加载训练数据，不用每次训练都重新生成array文件',
                    default=False)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='不训练embedding文件，一般来说加上这个参数效果不太好',
                    default=True)

parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)


parser.add_argument('--num_lstm', type=int,
                    help='LSTM层数个数，目前结果3层比5层好用，1层还在做实验中...',
                    default=1)

args = parser.parse_args()
# %%
NUM_creative_id = 2481135+1
NUM_ad_id = 2264190+1
NUM_product_id = 33273+1


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

    DATA = {}
    num_examples = Y_age.shape[0]
    train_examples = int(num_examples * 0.9)

    # 第一个输入
    # 获取 creative_id 特征
    X1_train, tokenizer = get_train(
        'creative_id', NUM_creative_id, LEN_creative_id)
    creative_id_emb = get_embedding('creative_id', tokenizer)

    DATA['X1_train'] = X1_train[:train_examples]
    DATA['X1_val'] = X1_train[train_examples:]
    DATA['creative_id_emb'] = creative_id_emb

    # 第二个输入
    # 获取 ad_id 特征
    X2_train, tokenizer = get_train(
        'ad_id', NUM_ad_id, LEN_ad_id)
    ad_id_emb = get_embedding('ad_id', tokenizer)

    DATA['X2_train'] = X2_train[:train_examples]
    DATA['X2_val'] = X2_train[train_examples:]
    DATA['ad_id_emb'] = ad_id_emb

    # 第三个输入
    # 获取 product_id 特征
    X3_train, tokenizer = get_train(
        'product_id', NUM_product_id, LEN_product_id)
    product_id_emb = get_embedding('product_id', tokenizer)

    DATA['X3_train'] = X3_train[:train_examples]
    DATA['X3_val'] = X3_train[train_examples:]
    DATA['product_id_emb'] = product_id_emb

    # 构造输出的训练标签
    # 获得age、gender标签
    user_train = pd.read_csv(
        'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
    Y_gender = user_train['gender'].values
    Y_age = user_train['age'].values
    Y_gender = Y_gender - 1
    Y_age = Y_age - 1
    Y_age = to_categorical(Y_age)
    Y_gender = to_categorical(Y_gender)

    DATA['Y_train'] = Y_age[:train_examples]
    DATA['Y_val'] = Y_age[train_examples:]

    # 分别对应 x1_train x1_val x2_train x2_val y_train y_val
    return DATA

# %%


def get_test():
    pass


# %%
LEN_creative_id = 100
LEN_ad_id = 100
LEN_product_id = 100


def get_model(DATA):
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id,
                   output_dim=128,
                   weights=[DATA['creative_id_emb']],
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
                   weights=[DATA['ad_id_emb']],
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
                   weights=[DATA['product_id_emb']],
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
    print('从csv文件提取训练数据到array格式，大概十分钟时间')
    DATA = get_train_val()
    mail('get train data done.')

    # 训练数据保存为npy文件
    dirs = 'tmp/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    def save_npy(datas, name):
        for i, data in enumerate(datas):
            np.save(f'tmp/{name}_{i}.npy', data)

    inputs = [x1_train, x1_val, x2_train, x2_val, x3_train, x3_val]
    targets = [y_train, y_val]
    embeddings = [creative_id_emb, ad_id_emb, product_id_emb]
    save_npy(inputs, 'inputs')
    save_npy(targets, 'age')
    save_npy(embeddings, 'embeddings')
else:
    DATA = {}
    DATA['X1_train'] = np.load('tmp/inputs_0.npy', allow_pickle=True)
    DATA['X1_val'] = np.load('tmp/inputs_1.npy', allow_pickle=True)
    DATA['X2_train'] = np.load('tmp/inputs_2.npy', allow_pickle=True)
    DATA['X2_val'] = np.load('tmp/inputs_3.npy', allow_pickle=True)
    DATA['X3_train'] = np.load('tmp/inputs_4.npy', allow_pickle=True)
    DATA['X3_val'] = np.load('tmp/inputs_5.npy', allow_pickle=True)
    DATA['Y_train'] = np.load('tmp/age_0.npy', allow_pickle=True)
    DATA['Y_val'] = np.load('tmp/age_1.npy', allow_pickle=True)
    DATA['creative_id_emb'] = np.load(
        'tmp/embeddings_0.npy', allow_pickle=True)
    DATA['ad_id_emb'] = np.load(
        'tmp/embeddings_1.npy', allow_pickle=True)
    DATA['product_id_emb'] = np.load(
        'tmp/embeddings_2.npy', allow_pickle=True)


# %%
model = get_model(DATA)
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
try:
    examples = args.examples
    mail('start train lstm')
    model.fit(
        {
            'creative_id': DATA['X1_train'][:examples],
            'ad_id': DATA['X2_train'][:examples],
            'product_id': DATA['X3_train'][:examples]
        },
        DATA['Y_train'][:examples],
        validation_data=([DATA['X1_val'], DATA['X2_val'],
                          DATA['X3_val']], DATA['Y_val']),
        epochs=args.epoch,
        batch_size=args.batch_size,
        callbacks=[checkpoint],
    )
    mail('train lstm done!!!')
except Exception as e:
    e = str(e)
    mail('train lstm failed!!! ' + e)


# %%
# 后续为预测过程，暂时注释掉不使用但是不要删除
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
