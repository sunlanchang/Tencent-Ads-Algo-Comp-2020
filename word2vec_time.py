# 通过用户访问的click_times的序列，生成每个click_times的词嵌入
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
from mymail import mail
# %%
df_train = pd.read_csv(
    'data/train_preliminary/clicklog_ad_user_train_eval_test.csv')
df_test = pd.read_csv('data/test/clicklog_ad_user_test.csv')
columns = ['user_id', 'click_times', 'time']
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
userid_click_timess = df_train_test_sorted.groupby(
    'user_id')['click_times'].apply(list).reset_index(name='click_timess')
# %%
with open('word2vec/userid_click_timess.txt', 'w')as f:
    for ids in userid_click_timess.click_timess:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
# %%
sentences = LineSentence('word2vec/userid_click_timess.txt')
dimension_embedding = 128
model = Word2Vec(sentences, size=dimension_embedding,
                 window=10, min_count=1, workers=-1, iter=10, sg=1)
model.save("word2vec/word2vec_click_times.model")
path = "word2vec/wordvectors_click_times.kv"
model.wv.save(path)
print('Save embedding done!!!')
# %%
path = "word2vec/wordvectors_click_times.kv"
wv = KeyedVectors.load(path, mmap='r')
dimension_embedding = 128
columns = ['c'+str(i) for i in range(dimension_embedding)]
data = {}
for col_name in columns:
    data[col_name] = pd.Series([], dtype='float')
df_click_times_embedding = pd.DataFrame(data)

# %%
data = {}
for key in tqdm(wv.vocab):
    data[int(key)] = wv[key].tolist()
# %%
df_click_times_embedding = pd.DataFrame.from_dict(
    data, orient='index',
    columns=columns)
df_click_times_embedding['click_times'] = df_click_times_embedding.index
# %%
df_click_times_embedding.to_hdf(
    'word2vec/df_click_times_embedding.h5',
    key='df_click_times_embedding', mode='w')
mail('save h5 done')
# %%
df_click_times_embedding = pd.read_hdf(
    'word2vec/df_click_times_embedding.h5',
    key='df_click_times_embedding', mode='r')
# %%
# %%
try:
    userid_click_times_embedding = pd.merge(
        df_train_test_sorted, df_click_times_embedding, on='click_times', how='left')
    userid_click_times_embedding.drop(
        columns=['click_times', 'time'], inplace=True)
    userid_click_times_embedding.groupby('user_id').mean().to_csv(
        'word2vec/click_times.csv', header=True, index=False)
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
# for user in tqdm(range(len(seq_click_times))):
#     user_em = df_click_times_embedding.loc[seq_click_times[user]].mean()
#     # df_user_embedding = df_user_embedding.append(user_em, ignore_index=True)
    # debug += 1
    # if debug == 10:
    #     break
# debug = 0
# frames = []
# for click_times in tqdm.tqdm(wv.vocab):
#     creativeid_embedding = wv[click_times]
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
# df_creativeid_embedding.iloc[-1, -1] = str(click_times)
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
