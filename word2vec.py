# 通过用户访问的creative_id的序列，生成每个creative_id的词嵌入
# %%
import pandas as pd
import numpy as np
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
# %%
df_train = pd.read_csv('data/train_preliminary/clicklog_ad_user.csv')
df_test = pd.read_csv('data/test/clicklog_ad.csv')
# %%
columns = ['user_id', 'creative_id', 'time']
frame = [df_train[columns], df_test[columns]]
df_train_test = pd.concat(frame, ignore_index=True)
# %%
df_train_test_sorted = df_train_test.sort_values(
    ["user_id", "time"], ascending=(True, True))
userid_creative_ids = df_train_test_sorted.groupby(
    'user_id')['time'].apply(list).reset_index(name='creative_ids')
# %%
with open('word2vec/userid_creativeids.txt', 'w')as f:
    for ids in userid_creative_ids.creative_ids:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
# %%
sentences = LineSentence('word2vec/userid_creativeids.txt')
model = Word2Vec(sentences, size=128, window=3, min_count=1, workers=-1)
model.save("word2vec/word2vec.model")
path = "word2vec/wordvectors.kv"
model.wv.save(path)
wv = KeyedVectors.load(path, mmap='r')
# %%
vector = wv['1']  # numpy vector of a word
