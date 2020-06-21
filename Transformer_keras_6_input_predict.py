# %%
# 生成词嵌入文件
from layers import Add, LayerNormalization
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import PositionEncoding
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from mymail import mail
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
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
from mymail import mail


tf.config.experimental_run_functions_eagerly(True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--gender', action='store_true',
                    help='gender model',
                    default=False)
parser.add_argument('--age', action='store_true',
                    help='age model',
                    default=False)

parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--predict', action='store_true',
                    help='从npy文件加载数据',
                    default=False)

parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--head_attention', type=int,
                    help='transformer head个数',
                    default=1)

parser.add_argument('--train_examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)
parser.add_argument('--val_examples', type=int,
                    help='验证集数据，调试时候可以设置1000',
                    default=90000)
args = parser.parse_args()
# %%
NUM_creative_id = 3412772
NUM_ad_id = 3027360
NUM_product_id = 39057
NUM_advertiser_id = 57870
NUM_industry = 332
NUM_product_category = 18

LEN_creative_id = 150
LEN_ad_id = 150
LEN_product_id = 150
LEN_advertiser_id = 150
LEN_industry = 150
LEN_product_category = 150

# %%


def get_gender_model(DATA):

    feed_forward_size = 2048
    max_seq_len = 100
    model_dim = 128*6

    input_creative_id = Input(shape=(max_seq_len,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
                   output_dim=128,
                   weights=[DATA['creative_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_creative_id)
    # encodings = PositionEncoding(model_dim)(x1)
    # encodings = Add()([embeddings, encodings])

    input_ad_id = Input(shape=(max_seq_len,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id+1,
                   output_dim=128,
                   weights=[DATA['ad_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_ad_id)

    input_product_id = Input(shape=(max_seq_len,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=128,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=128,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=128,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=128,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=100,
                   mask_zero=True)(input_product_category)

    # (bs, 100, 128*2)
    encodings = layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6])
    # (bs, 100)
    masks = tf.equal(input_creative_id, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(8, 96)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim, feed_forward_size)
    ff_out = ff(attention_out)
    # Add & Norm
    # ff_out (bs, 100, 128)，但是attention_out是(bs,100,256)
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_gender = Dense(2, activation='softmax', name='gender')(encodings)
    # output_age = Dense(10, activation='softmax', name='age')(encodings)

    model = Model(
        inputs=[input_creative_id,
                input_ad_id,
                input_product_id,
                input_advertiser_id,
                input_industry,
                input_product_category],
        outputs=[output_gender]
    )

    model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={
            'gender': losses.CategoricalCrossentropy(from_logits=False),
            # 'age': losses.CategoricalCrossentropy(from_logits=False)
        },
        # loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    return model


def get_age_model(DATA):

    feed_forward_size = 2048
    max_seq_len = 150
    model_dim = 256+256+64+32+8+16

    input_creative_id = Input(shape=(max_seq_len,), name='creative_id')
    x1 = Embedding(input_dim=NUM_creative_id+1,
                   output_dim=256,
                   weights=[DATA['creative_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_creative_id)
    # encodings = PositionEncoding(model_dim)(x1)
    # encodings = Add()([embeddings, encodings])

    input_ad_id = Input(shape=(max_seq_len,), name='ad_id')
    x2 = Embedding(input_dim=NUM_ad_id+1,
                   output_dim=256,
                   weights=[DATA['ad_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_ad_id)

    input_product_id = Input(shape=(max_seq_len,), name='product_id')
    x3 = Embedding(input_dim=NUM_product_id+1,
                   output_dim=32,
                   weights=[DATA['product_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_product_id)

    input_advertiser_id = Input(shape=(max_seq_len,), name='advertiser_id')
    x4 = Embedding(input_dim=NUM_advertiser_id+1,
                   output_dim=64,
                   weights=[DATA['advertiser_id_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_advertiser_id)

    input_industry = Input(shape=(max_seq_len,), name='industry')
    x5 = Embedding(input_dim=NUM_industry+1,
                   output_dim=16,
                   weights=[DATA['industry_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_industry)

    input_product_category = Input(
        shape=(max_seq_len,), name='product_category')
    x6 = Embedding(input_dim=NUM_product_category+1,
                   output_dim=8,
                   weights=[DATA['product_category_emb']],
                   trainable=args.not_train_embedding,
                   #    trainable=False,
                   input_length=max_seq_len,
                   mask_zero=True)(input_product_category)

    # (bs, 100, 128*2)
    encodings = layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6])
    # (bs, 100)
    masks = tf.equal(input_creative_id, 0)

    # (bs, 100, 128*2)
    # concat之后是632
    attention_out = MultiHeadAttention(8, 79)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim, feed_forward_size)
    ff_out = ff(attention_out)
    # Add & Norm
    # ff_out (bs, 100, 128)，但是attention_out是(bs,100,256)
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    # output_gender = Dense(2, activation='softmax', name='gender')(encodings)
    output_age = Dense(10, activation='softmax', name='age')(encodings)

    model = Model(
        inputs=[input_creative_id,
                input_ad_id,
                input_product_id,
                input_advertiser_id,
                input_industry,
                input_product_category],
        outputs=[output_age]
    )

    model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={
            # 'gender': losses.CategoricalCrossentropy(from_logits=False),
            'age': losses.CategoricalCrossentropy(from_logits=False)
        },
        # loss_weights=[0.4, 0.6],
        metrics=['accuracy'])
    return model


def get_train_val():

    # 从序列文件提取array格式数据
    def get_train(feature_name, vocab_size, len_feature):
        ########################################
        f = open(f'word2vec_new/{feature_name}.txt')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(f)
        f.close()

        feature_seq = []
        #########################################
        with open(f'word2vec_new/{feature_name}.txt') as f:
            for text in f:
                feature_seq.append(text.strip())

        sequences = tokenizer.texts_to_sequences(feature_seq[:900000])
        X_train = pad_sequences(
            sequences, maxlen=len_feature, padding='post')

        sequences = tokenizer.texts_to_sequences(feature_seq[900000:])
        X_test = pad_sequences(
            sequences, maxlen=len_feature, padding='post')
        return X_train, tokenizer, X_test

    # 提取词向量文件
    def get_embedding(feature_name, tokenizer):
        ########################################
        path = f'word2vec_new/{feature_name}.kv'
        wv = KeyedVectors.load(path, mmap='r')
        feature_tokens = list(wv.vocab.keys())
        feature_name_dict = {'creative_id': 256, 'ad_id': 256, 'advertiser_id': 64,
                             'product_id': 32, 'product_category': 8, 'industry': 16}
        embedding_dim = feature_name_dict[feature_name]
        embedding_matrix = np.random.randn(
            len(feature_tokens)+1, embedding_dim)
        for word, i in tokenizer.word_index.items():
            embedding_vector = wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print(str(word)+' 没有找到')
        return embedding_matrix

    DATA = {}
    # 获取test数据

    # 构造输出的训练标签
    # 获得age、gender标签
    #######################################################
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
    X1_train, tokenizer, X1_test = get_train(
        'creative_id', NUM_creative_id+1, LEN_creative_id)  # +1为了UNK的creative_id
    creative_id_emb = get_embedding('creative_id', tokenizer)

    DATA['X1_train'] = X1_train[:train_examples]
    DATA['X1_val'] = X1_train[train_examples:]
    DATA['X1_test'] = X1_test
    DATA['creative_id_emb'] = creative_id_emb

    # 第二个输入
    print('获取 ad_id 特征')
    X2_train, tokenizer, X2_test = get_train(
        'ad_id', NUM_ad_id+1, LEN_ad_id)
    ad_id_emb = get_embedding('ad_id', tokenizer)

    DATA['X2_train'] = X2_train[:train_examples]
    DATA['X2_val'] = X2_train[train_examples:]
    DATA['X2_test'] = X2_test

    DATA['ad_id_emb'] = ad_id_emb

    # 第三个输入
    print('获取 product_id 特征')
    X3_train, tokenizer, X3_test = get_train(
        'product_id', NUM_product_id+1, LEN_product_id)
    product_id_emb = get_embedding('product_id', tokenizer)

    DATA['X3_train'] = X3_train[:train_examples]
    DATA['X3_val'] = X3_train[train_examples:]
    DATA['X3_test'] = X3_test
    DATA['product_id_emb'] = product_id_emb

    # 第四个输入
    print('获取 advertiser_id 特征')
    X4_train, tokenizer, X4_test = get_train(
        'advertiser_id', NUM_advertiser_id+1, LEN_advertiser_id)
    advertiser_id_emb = get_embedding('advertiser_id', tokenizer)

    DATA['X4_train'] = X4_train[:train_examples]
    DATA['X4_val'] = X4_train[train_examples:]
    DATA['X4_test'] = X4_test
    DATA['advertiser_id_emb'] = advertiser_id_emb

    # 第五个输入
    print('获取 industry 特征')
    X5_train, tokenizer, X5_test = get_train(
        'industry', NUM_industry+1, LEN_industry)
    industry_emb = get_embedding('industry', tokenizer)

    DATA['X5_train'] = X5_train[:train_examples]
    DATA['X5_val'] = X5_train[train_examples:]
    DATA['X5_test'] = X5_test
    DATA['industry_emb'] = industry_emb

    # 第六个输入
    print('获取 product_category 特征')
    X6_train, tokenizer, X6_test = get_train(
        'product_category', NUM_product_category+1, LEN_product_category)
    product_category_emb = get_embedding('product_category', tokenizer)

    DATA['X6_train'] = X6_train[:train_examples]
    DATA['X6_val'] = X6_train[train_examples:]
    DATA['X6_test'] = X6_test
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

    test = [DATA['X1_test'],
            DATA['X2_test'],
            DATA['X3_test'],
            DATA['X4_test'],
            DATA['X5_test'],
            DATA['X6_test'], ]
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
    save_npy(test, 'test')
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

    DATA['X_test1'] = np.load('tmp/test_0.npy', allow_pickle=True)
    DATA['X_test2'] = np.load('tmp/test_1.npy', allow_pickle=True)
    DATA['X_test3'] = np.load('tmp/test_2.npy', allow_pickle=True)
    DATA['X_test4'] = np.load('tmp/test_3.npy', allow_pickle=True)
    DATA['X_test5'] = np.load('tmp/test_4.npy', allow_pickle=True)
    DATA['X_test6'] = np.load('tmp/test_5.npy', allow_pickle=True)


# %%

# # %%
# %%
if args.gender:
    model = get_gender_model(DATA)
if args.age:
    model = get_age_model(DATA)

##############################################
model.load_weights('tmp/gender_epoch_01.hdf5')

y_pred = model.predict(
    {
        'creative_id': DATA['X1_test'],
        'ad_id': DATA['X2_test'],
        'product_id': DATA['X3_test'],
        'advertiser_id': DATA['X4_test'],
        'industry': DATA['X5_test'],
        'product_category': DATA['X6_test']
    },
    batch_size=1024,
)
y_pred = np.argmax(y_pred, axis=1)
y_pred = y_pred.flatten()
y_pred += 1

if args.gender:
    ans = pd.DataFrame({'predicted_gender': y_pred})
    ################################################
    ans.to_csv(
        'data/ans/transformer_gender.csv', header=True, columns=['predicted_gender'], index=False)
elif args.age:
    ans = pd.DataFrame({'predicted_age': y_pred})
    ################################################
    ans.to_csv(
        'data/ans/transformer_age.csv', header=True, columns=['predicted_age'], index=False)

    ##############################################
    user_id_test = pd.read_csv(
        'data/test/clicklog_ad.csv').sort_values(['user_id'], ascending=(True,)).user_id.unique()
    ans = pd.DataFrame({'user_id': user_id_test})

    ##############################################
    gender = pd.read_csv('data/ans/transformer_gender.csv')
    age = pd.read_csv('data/ans/transformer_age.csv')
    ans['predicted_gender'] = gender.predicted_gender
    ans['predicted_age'] = age.predicted_age
    ans.to_csv('data/ans/submission.csv', header=True, index=False,
               columns=['user_id', 'predicted_age', 'predicted_gender'])
