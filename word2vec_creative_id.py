# 通过用户访问的creative_id的序列，生成每个creative_id的词嵌入
# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import pickle
from mail import mail
# %%
df_train = pd.read_csv(
    'data/train_preliminary/clicklog_ad_user_train_eval_test.csv')
df_test = pd.read_csv('data/test/clicklog_ad_user_test.csv')
columns = ['user_id', 'creative_id', 'time']
frame = [df_train[columns], df_test[columns]]
df_train_test = pd.concat(frame, ignore_index=True)
df_train_test_sorted = df_train_test.sort_values(
    ["user_id", "time"], ascending=(True, True))
# %%
with open('word2vec/df_train_test_sorted.pkl', 'wb') as f:
    pickle.dump(df_train_test_sorted, f)
# %%
with open('word2vec/df_train_test_sorted.pkl', 'rb') as f:
    df_train_test_sorted = pickle.load(f)
# %%
userid_creative_ids = df_train_test_sorted.groupby(
    'user_id')['creative_id'].apply(list).reset_index(name='creative_ids')
# %%
with open('word2vec/userid_creative_ids.txt', 'w')as f:
    for ids in userid_creative_ids.creative_ids:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
# %%
sentences = LineSentence('word2vec/userid_creative_ids.txt')
dimension_embedding = 128
model = Word2Vec(sentences, size=dimension_embedding,
                 window=10, min_count=1, workers=-1, iter=10, sg=1)
model.save("word2vec/word2vec_creative_id.model")
path = "word2vec/wordvectors_creative_id.kv"
model.wv.save(path)
print('Save embedding done!!!')
# %%
path = "word2vec/wordvectors_creative_id.kv"
wv = KeyedVectors.load(path, mmap='r')
dimension_embedding = 128
columns = ['c'+str(i) for i in range(dimension_embedding)]
data = {}
for col_name in columns:
    data[col_name] = pd.Series([], dtype='float')
df_creative_id_embedding = pd.DataFrame(data)

# %%
data = {}
for key in tqdm(wv.vocab):
    data[int(key)] = wv[key].tolist()
# %%
df_creative_id_embedding = pd.DataFrame.from_dict(
    data, orient='index',
    columns=columns)
df_creative_id_embedding['creative_id'] = df_creative_id_embedding.index
# %%
df_creative_id_embedding.to_hdf(
    'word2vec/df_creative_id_embedding.h5',
    key='df_creative_id_embedding', mode='w')
mail('save h5 done')
# %%
df_creative_id_embedding = pd.read_hdf(
    'word2vec/df_creative_id_embedding.h5',
    key='df_creative_id_embedding', mode='r')
# %%
# %%
try:
    userid_creative_id_embedding = pd.merge(
        df_train_test_sorted, df_creative_id_embedding, on='creative_id', how='left')
    userid_creative_id_embedding.drop(
        columns=['creative_id', 'time'], inplace=True)
    userid_creative_id_embedding.groupby('user_id').mean().to_csv(
        'word2vec/creative_id.csv', header=True, index=False)
    mail('to csv done')
except:
    mail('failed')
# %%
# columns = ['c'+str(i) for i in range(128)]
# data = {}
# for col_name in columns:
#     data[col_name] = pd.Series([], dtype='float')
# df_user_embedding = pd.DataFrame(data)
# # %%
# # this will take 24 hours!!!
# # debug = 0
# for user in tqdm(range(len(seq_creative_id))):
#     user_em = df_creative_id_embedding.loc[seq_creative_id[user]].mean()
#     # df_user_embedding = df_user_embedding.append(user_em, ignore_index=True)
    # debug += 1
    # if debug == 10:
    #     break
# debug = 0
# frames = []
# for creative_id in tqdm.tqdm(wv.vocab):
#     creativeid_embedding = wv[creative_id]
#     tmp = pd.DataFrame(
#         creativeid_embedding.reshape(-1, len(creativeid_embedding)),
#         columns=columns[:-1])
#     # df_creativeid_embedding = df_creativeid_embedding.append(tmp)
#     frames.append(tmp)
#     if len(frames) == 1000000:
#         # frames = [df_creativeid_embedding, tmp]
#         frames = [df_creativeid_embedding]+frames
#         df_creativeid_embedding = pd.concat(frames)
#         frames = []
# df_creativeid_embedding.iloc[-1, -1] = str(creative_id)
# %%
# if len(frames) != 0:
#     frames = [df_creativeid_embedding]+frames
#     df_creativeid_embedding = pd.concat(frames)
# df_creativeid_embedding.to_hdf('data/clicklog_ad_user_train_eval_test.h5',
#                                key='df_creativeid_embedding', mode='w')

# debug += 1
# if debug == 10:
#     break


# %%
