# %%
# 生成词嵌入文件
from tqdm import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from mail import mail
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%

debug = True

# %%
if debug:
    # f = open('tmp/userid_creative_ids.txt')
    f = open('word2vec/userid_creative_ids.txt')
else:
    pass
num_creative_id = 2481135+1
tokenizer = Tokenizer(num_words=num_creative_id)
tokenizer.fit_on_texts(f)
f.close()


# %%
path = "word2vec/wordvectors_creative_id.kv"
wv = KeyedVectors.load(path, mmap='r')


# %%
# f = open('word2vec/userid_creative_ids.txt')
# max_len_creative_id = -1
# for line in f:
#     current_line_len = len(line.strip().split(' '))
#     max_len_creative_id = max(max_len_creative_id, current_line_len)
# f.close()


# %%
creative_id_tokens = list(wv.vocab.keys())
embedding_dim = 128
embedding_matrix = np.random.randn(len(creative_id_tokens)+1, 128)
cnt = 0
for creative_id in creative_id_tokens:
    embedding_vector = wv[creative_id]
    if embedding_vector is not None:
        index = tokenizer.texts_to_sequences([creative_id])[0][0]
        embedding_matrix[index] = embedding_vector


# %%
if debug:
    max_len_creative_id = 100
# shape：(sequence长度, )
input_x = Input(shape=(None,))
x = Embedding(input_dim=num_creative_id,
              output_dim=128,
              weights=[embedding_matrix],
              trainable=True,
              input_length=max_len_creative_id,
              mask_zero=True)(input_x)
x = LSTM(1024, return_sequences=True)(x)
x = LSTM(512, return_sequences=False)(x)
x = Dense(128)(x)
x = Dropout(0.5)(x)
output_y = Dense(1, activation='sigmoid')(x)

model = Model([input_x], output_y)

# 这种方式构建模型灵活性差但是方便构建
# model = Sequential([
#     Embedding(num_creative_id, 128,
#               weights=[embedding_matrix],
#               trainable=False,
#               input_length=None),
#     LSTM(1024),
#     Dense(1, activation='sigmoid')
# ])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# %%
# 测试数据格式(batch_size, sequence长度)
test_data = np.array([1, 2, 3, 4]).reshape(1, -1)
model.predict(test_data)


# %%
creative_id_seq = []
with open('word2vec/userid_creative_ids.txt') as f:
    for text in f:
        creative_id_seq.append(text.strip())


# %%
if debug:
    sequences = tokenizer.texts_to_sequences(creative_id_seq[:900000//1])
else:
    sequences = tokenizer.texts_to_sequences(creative_id_seq)

X_train = pad_sequences(sequences, maxlen=max_len_creative_id, padding='post')

# %%
user_train = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_gender = user_train['gender'].values
Y_age = user_train['age'].values

Y_gender = Y_gender - 1
# %%
if debug:
    Y_gender = Y_gender[:900000//100]
# %%
checkpoint = ModelCheckpoint("tmp/gender_epoch_{epoch:02d}.hdf5", monitor='val_loss', verbose=0,
                             save_best_only=False, mode='auto', period=1)

# %%
try:
    mail('start train lstm')
    model.fit(X_train,
              Y_gender,
              validation_split=0.1,
              epochs=100,
              batch_size=768,
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

X_test = pad_sequences(sequences, maxlen=max_len_creative_id)
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
