# 通过用户访问的creative_id的序列，生成每个creative_id的词嵌入
# %%
import pandas as pd
import numpy as np
import tqdm
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
# %%
df_train = pd.read_csv(
    'data/train_preliminary/clicklog_ad_user_train_eval_test.csv')
df_test = pd.read_csv('data/test/clicklog_ad_user_test.csv')
columns = ['user_id', 'creative_id', 'time']
frame = [df_train[columns], df_test[columns]]
df_train_test = pd.concat(frame, ignore_index=True)
# %%
df_train_test_sorted = df_train_test.sort_values(
    ["user_id", "time"], ascending=(True, True))
userid_creative_ids = df_train_test_sorted.groupby(
    'user_id')['creative_id'].apply(list).reset_index(name='creative_ids')
# %%
with open('word2vec/userid_creativeids.txt', 'w')as f:
    for ids in userid_creative_ids.creative_ids:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
# %%
sentences = LineSentence('word2vec/userid_creativeids.txt')
dimension_embedding = 128
model = Word2Vec(sentences, size=dimension_embedding,
                 window=3, min_count=1, workers=-1)
model.save("word2vec/word2vec.model")
path = "word2vec/wordvectors.kv"
model.wv.save(path)
print('Save embedding done!!!')
# %%
path = "word2vec/wordvectors.kv"
wv = KeyedVectors.load(path, mmap='r')
columns = ['c'+str(i) for i in range(128)]
data = {}
for col_name in columns:
    data[col_name] = pd.Series([], dtype='float')
df_creativeid_embedding = pd.DataFrame(data)

# %%
data = {}
for key in tqdm.tqdm(wv.vocab):
    data[key] = wv[key].tolist()
# %%

df_creativeid_embedding = pd.DataFrame.from_dict(
    data, orient='index',
    columns=columns)
# %%
df_creativeid_embedding.to_hdf(
    'word2vec/df_creativeid_embedding.h5',
    key='df_creativeid_embedding', mode='w')
# %%
df_creativeid_embedding = pd.read_hdf(
    'word2vec/df_creativeid_embedding.h5',
    key='df_creativeid_embedding', mode='r')
# %%
with open('word2vec/userid_creativeids.txt', 'r')as f:
    seq_creative_id = f.readlines()
seq_creative_id = [[str(e) for e in line.strip().split(' ')]
                   for line in seq_creative_id]
# %%
columns = ['c'+str(i) for i in range(128)]
data = {}
for col_name in columns:
    data[col_name] = pd.Series([], dtype='float')
user_embedding = pd.DataFrame(data)
# %%
for user in tqdm.tqdm(range(len(seq_creative_id))):
    df_creativeid_embedding.loc[seq_creative_id[user], :].mean()
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
