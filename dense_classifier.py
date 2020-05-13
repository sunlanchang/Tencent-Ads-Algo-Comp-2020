# %%
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tqdm
from tqdm import tnrange, tqdm_notebook

# %%
# 合成用户embedding
path = "word2vec/wordvectors.kv"
wv = KeyedVectors.load(path, mmap='r')
with open('word2vec/userid_creativeids.txt', 'r')as f:
    lines = f.readlines()
lines = [[int(e) for e in line.split(' ')] for line in lines]
number_train_user = 900000
number_test_user = 1000000
user_train = lines[:number_train_user]
user_test = lines[number_train_user:]
columns = ['c'+str(i) for i in range(128)]
data = {}
for col_name in columns:
    data[col_name] = pd.Series([], dtype='float')
df_user_train = pd.DataFrame(data)
df_user_test = pd.DataFrame(data)
# %%
for line in tqdm.tqdm(user_train):
    user_embedding_train = np.zeros(128)
    for creative_id in line:
        user_embedding_train += wv[str(creative_id)]
    user_embedding_train = user_embedding_train / len(line)
    tmp = pd.DataFrame(user_embedding_train.reshape(-1,
                                                    len(user_embedding_train)), columns=columns)
    df_user_train = df_user_train.append(tmp)
# %%
for line in tqdm.tqdm(user_test):
    user_embedding_test = np.zeros(128)
    for creative_id in line:
        user_embedding_test += wv[str(creative_id)]
    user_embedding_test = user_embedding_test / len(line)
    tmp = pd.DataFrame(user_embedding_test.reshape(-1,
                                                   len(user_embedding_train)), columns=columns)
    df_user_test = df_user_test.append(tmp)
# %%
# 将同一个用户creative_id相加平均后即为一个用户的Embedding
all_train_data = pd.read_csv(
    'data/train_preliminary/clicklog_ad_user_train_eval_test.csv')
all_train_data = all_train_data.sort_values(
    ["user_id"], ascending=(True))
# %%
all_test_data = pd.read_csv(
    'data/test/clicklog_ad_user_test.csv')
all_test_data = all_test_data.sort_values(
    ["user_id"], ascending=(True))
# %%
assert df_user_train.shape[0] == all_train_data.shape[0]
df_user_train['user_id'] = all_train_data['user_id']
df_user_train['gender'] = all_train_data['gender']
df_user_train['age'] = all_train_data['age']
df_user_train.to_hdf('word2vec/df_user_train_test.h5',
                     key='df_user_train', mode='w')
# %%
assert df_user_test.shape[0] == all_test_data.shape[0]
df_user_test['user_id'] = all_test_data['user_id']
df_user_test.to_hdf('word2vec/df_user_train_test.h5',
                    key='df_user_test', mode='a')
