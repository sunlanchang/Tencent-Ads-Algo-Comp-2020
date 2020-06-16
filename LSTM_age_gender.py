# %%
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Embedding, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec, KeyedVectors
from mymail import mail
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
先进LSTM再concat
python LSTM_age_gender.py --epoch 3 --batch_size 128 --train_examples 810000 --val_examples 90000 --num_lstm 3 --tail_concat

先concat再进LSTM
python LSTM_age_gender.py --epoch 3 --batch_size 128 --train_examples 810000 --val_examples 90000 --num_lstm 3 --head_concat
'''

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
parser.add_argument('--train_examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)
parser.add_argument('--val_examples', type=int,
                    help='验证集数据，调试时候可以设置1000',
                    default=90000)


parser.add_argument('--num_lstm', type=int,
                    help='LSTM层数个数，目前结果3层比5层好用，1层还在做实验中...',
                    default=1)
parser.add_argument('--head_concat', action='store_true',
                    help='从npy文件加载训练数据，不用每次训练都重新生成array文件',
                    default=False)
parser.add_argument('--tail_concat', action='store_true',
                    help='从npy文件加载训练数据，不用每次训练都重新生成array文件',
                    default=False)

args = parser.parse_args()
# %%
NUM_creative_id = 2481135
NUM_ad_id = 2264190
NUM_product_id = 33273
NUM_advertiser_id = 52090
NUM_industry = 326
NUM_product_category = 18


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


def get_test():
    pass


# %%
LEN_creative_id = 100
LEN_ad_id = 100
LEN_product_id = 100
LEN_advertiser_id = 100
LEN_industry = 100
LEN_product_category = 100


def get_tail_concat_model(DATA):
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
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
    x2 = Embedding(input_dim=NUM_ad_id+1,
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
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=128,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_product_id,
                   mask_zero=True)(input_product_id)
    for _ in range(args.num_lstm):
        x3 = Bidirectional(LSTM(256, return_sequences=True))(x3)
    x3 = layers.GlobalMaxPooling1D()(x3)

    # third input
    input_advertiser_id = Input(shape=(None,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=128,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_advertiser_id,
                   mask_zero=True)(input_advertiser_id)
    for _ in range(args.num_lstm):
        x4 = Bidirectional(LSTM(256, return_sequences=True))(x4)
    x4 = layers.GlobalMaxPooling1D()(x4)

    # third input
    input_industry = Input(shape=(None,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=128,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_industry,
                   mask_zero=True)(input_industry)
    for _ in range(args.num_lstm):
        x5 = Bidirectional(LSTM(256, return_sequences=True))(x5)
    x5 = layers.GlobalMaxPooling1D()(x5)

    # third input
    input_product_category = Input(shape=(None,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=128,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_product_category,
                   mask_zero=True)(input_product_category)
    for _ in range(args.num_lstm):
        x6 = Bidirectional(LSTM(256, return_sequences=True))(x6)
    x6 = layers.GlobalMaxPooling1D()(x6)

    x = layers.Concatenate(axis=1)([x1, x2, x3, x4, x5, x6])
    output_y = Dense(10, activation='softmax', name='age')(x)

    model = Model(
        [
            input_creative_id, input_ad_id, input_product_id,
            input_advertiser_id, input_industry, input_product_category
        ],
        output_y)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

# %%


def get_head_concat_model(DATA):
    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
                   output_dim=128,
                   weights=[DATA['creative_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_creative_id,
                   mask_zero=True)(input_creative_id)

    input_ad_id = Input(shape=(None,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id+1,
                   output_dim=128,
                   weights=[DATA['ad_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_ad_id,
                   mask_zero=True)(input_ad_id)

    input_product_id = Input(shape=(None,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=128,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_product_id,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(None,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=128,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_advertiser_id,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(None,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=128,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_industry,
                   mask_zero=True)(input_industry)

    input_product_category = Input(shape=(None,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=128,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,
                   input_length=LEN_product_category,
                   mask_zero=True)(input_product_category)

    x = Concatenate(axis=1)([x1, x2, x3, x4, x5, x6])

    for _ in range(args.num_lstm):
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    # x = layers.GlobalAvaregePooling1D()(x)

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
# model = get_model(DATA)
if args.head_concat:
    model = get_head_concat_model(DATA)
elif args.tail_concat:
    model = get_tail_concat_model(DATA)
# %%
# %%
# 测试数据格式(batch_size, sequence长度)
# x1 = np.array([1, 2, 3, 4]).reshape(1, -1)
# x2 = np.array([1, 2, 3, 4]).reshape(1, -1)
# model.predict([x1, x2])


# %%
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


lr = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint("tmp/lstm_epoch_{epoch:02d}.hdf5", monitor='val_loss', verbose=1,
                             save_best_only=False, mode='auto', period=1)
# %%
try:
    train_examples = args.train_examples
    val_examples = args.val_examples
    mail('start train lstm')
    if args.head_concat:
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
    elif args.tail_concat:
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
