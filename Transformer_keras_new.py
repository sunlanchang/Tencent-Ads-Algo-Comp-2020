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

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from layers import PositionEncoding
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import Add, LayerNormalization

tf.config.experimental_run_functions_eagerly(True)

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


class Transformer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, model_dim,
                 n_heads=8, encoder_stack=6, decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        super(Transformer, self).build(input_shape)

    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5  # Scale

        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)

        # Embedings + Postion-encodings
        # encodings = embeddings + position_encodings
        encodings = embeddings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = MultiHeadAttention(
                self._n_heads, self._model_dim // self._n_heads)
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = LayerNormalization()(attention_out)
            # Feed-Forward
            ff = PositionWiseFeedForward(
                self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = MultiHeadAttention(
                self._n_heads, self._model_dim // self._n_heads, future=True)
            masked_attention_input = [encodings,
                                      encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = LayerNormalization()(masked_attention_out)

            # Multi-head-Attention
            attention = MultiHeadAttention(
                self._n_heads, self._model_dim // self._n_heads)
            attention_input = [masked_attention_out,
                               encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = LayerNormalization()(attention_out)

            # Feed-Forward
            ff = PositionWiseFeedForward(
                self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        # Pre-Softmax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder(
            [decoder_inputs, encoder_encodings, encoder_masks])
        # return encoder_outputs
        return encoder_encodings

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self._vocab_size)


class Noam(Callback):

    def __init__(self, model_dim, step_num=0, warmup_steps=4000, verbose=False, **kwargs):
        self._model_dim = model_dim
        self._step_num = step_num
        self._warmup_steps = warmup_steps
        self.verbose = verbose
        super(Noam, self).__init__(**kwargs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        init_lr = self._model_dim ** -.5 * self._warmup_steps ** -1.5
        K.set_value(self.model.optimizer.lr, init_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self._step_num += 1
        lrate = self._model_dim ** -.5 * \
            K.minimum(self._step_num ** -.5, self._step_num *
                      self._warmup_steps ** -1.5)
        K.set_value(self.model.optimizer.lr, lrate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            lrate = K.get_value(self.model.optimizer.lr)
            print(f"epoch {epoch} lr: {lrate}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def label_smoothing(inputs, epsilon=0.1):

    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


# %%
NUM_creative_id = 3412772
NUM_ad_id = 3027360
NUM_product_id = 39057
NUM_advertiser_id = 57870
NUM_industry = 332
NUM_product_category = 18

vocab_size = 5000
max_seq_len = 100
model_dim = 128

input_creative_id = Input(shape=(max_seq_len,), name='encoder_inputs')
decoder_inputs = Input(shape=(max_seq_len,), name='input_creative_id')

X1 = Transformer(NUM_creative_id, model_dim)([encoder_inputs, decoder_inputs])

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

model.summary()


# %%

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
        path = f'word2vec_new/{feature_name}.kv'
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
        f = open(f'word2vec_new/{feature_name}.txt')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(f)
        f.close()

        feature_seq = []
        with open(f'word2vec_new/{feature_name}.txt') as f:
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
