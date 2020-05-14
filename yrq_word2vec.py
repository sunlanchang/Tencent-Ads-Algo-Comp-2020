# %%
import numpy as np
import pandas as pd
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import re
from tqdm import tqdm

import sys
import time

# Data_Root
# raw
train_raw_data_root = 'data/train_preliminary'
test_raw_data_root = 'data/test'

# Env-CSV_Data
# train
train_ad_filepath = train_raw_data_root + '/ad.csv'
train_click_log_filepath = train_raw_data_root + '/click_log.csv'
train_user_filepath = train_raw_data_root + '/user.csv'
# test
test_ad_filepath = test_raw_data_root + '/ad.csv'
test_click_log_filepath = test_raw_data_root + '/click_log.csv'

# word2vec
word2vec_dict_filepath = 'word2vec/dict.txt'
word2vec_word2vec_model_filepath = 'word2vec/word2vec.model'
word2vec_wordvectors_kv_filepath = 'word2vec/wordvectors.kv'
data_vec_filepath = 'word2vec/data_vec.csv'


def data():
    train_ad = pd.read_csv(train_ad_filepath)
    print('train_ad Read Done')
    train_click_log = pd.read_csv(train_click_log_filepath)
    print('train_click_log Read Done')
    train_user = pd.read_csv(train_user_filepath)
    print('train_user Read Done')

    test_ad = pd.read_csv(test_ad_filepath)
    print('test_ad Read Done')
    test_click_log = pd.read_csv(test_click_log_filepath)
    print('test_click_log Read Done')
    print('\nData Read Done\n')
    return train_ad, train_click_log, train_user, test_ad, test_click_log


# %%
train_ad, train_click_log, train_user, test_ad, test_click_log = data()

click_log = train_click_log.append(test_click_log)
ad = train_ad.append(test_ad)
data = pd.merge(click_log, ad, on='creative_id', how='left').fillna(
    int(-1)).replace('\\N', int(-1)).astype(int)
data_creativeid = data.groupby("user_id")['creative_id'].apply(
    list).reset_index(name='creative_id')
# data_product_category = data.groupby("user_id")['product_category'].apply(
#     list).reset_index(name='product_category')
# dict = pd.merge(data_industry, data_product_category,
#                 on='user_id', how='inner')
# product_category_tmp = []
# with tqdm(total=int(len(data_creativeid))) as pbar:
#     for j in dict["product_category"]:
#         product_category_tmp.append([i+400 for i in j])
#         pbar.update(1)
# tmp = pd.Series(product_category_tmp)

# dict["product_category_industry"] = tmp + \
#     dict["industry"]  # .map(str) product_category+400

#dict.drop(labels='user_id',axis=1).to_csv(word2vec_dict_filepath, index=False, header=False)
# with open(word2vec_dict_filepath, 'w')as f:
#     with tqdm(total=int(len(dict['product_category_industry']))) as pbar:
#         for i in dict['product_category_industry']:
#             i = [str(e) for e in i]
#             line = ' '.join(i)
#             f.write(line+'\n')
#             pbar.update(1)
# sentences = LineSentence(word2vec_dict_filepath)
# dimension_embedding = 32
# model = Word2Vec(sentences, size=dimension_embedding,
#                  window=3, min_count=1, workers=-1)
# model.save(word2vec_word2vec_model_filepath)
# model.wv.save(word2vec_wordvectors_kv_filepath)
# %%
word2vec_wordvectors_kv_filepath = 'word2vec/wordvectors.kv'
wv = KeyedVectors.load(word2vec_wordvectors_kv_filepath, mmap='r')

dict_embd_creativeid = {}
with tqdm(total=int(len(wv.vocab))) as pbar:
    for key in wv.vocab:
        dict_embd_creativeid[key] = wv[key].tolist()
        pbar.update(1)
# %%
product_category_industry_tmp = pd.Series(
    list(dict_embd_creativeid.keys())).astype(int)
vec_tmp = pd.Series(list(dict_embd_creativeid.values()))
embd_product_category_industry_pd = pd.DataFrame(
    columns=['product_category_industry', 'vec'])
embd_product_category_industry_pd['product_category_industry'] = product_category_industry_tmp
embd_product_category_industry_pd['vec'] = vec_tmp

data_vec_product_category_tmp = pd.DataFrame(data, columns=['user_id', 'product_category']).rename(
    {'product_category': 'product_category_industry'}, axis='columns')
data_vec_industry_tmp = pd.DataFrame(data, columns=['user_id', 'industry']).rename(
    {'industry': 'product_category_industry'}, axis='columns')
data_vec_product_category_tmp['product_category_industry'] = data_vec_product_category_tmp['product_category_industry']+400

data_vec_product_category = pd.merge(
    data_vec_product_category_tmp, embd_product_category_industry_pd, on='product_category_industry', how='left')
data_vec_industry = pd.merge(
    data_vec_industry_tmp, embd_product_category_industry_pd, on='product_category_industry', how='left')

data_vec_tmp = pd.concat(
    [data_vec_product_category, data_vec_industry], names=['user_id', 'vec'])
data_vec_list_tmp_pd = pd.DataFrame(pd.DataFrame(
    data_vec_tmp, columns=['user_id', 'vec'])['vec'].tolist())
data_user_id_list_tmp_pd = pd.DataFrame(pd.DataFrame(
    data_vec_tmp, columns=['user_id', 'vec'])['user_id'].tolist())
data_vec_tmp = pd.merge(data_user_id_list_tmp_pd, data_vec_list_tmp_pd, left_index=True,
                        right_index=True).rename({'0_x': 'user_id', '0_y': '0'}, axis='columns')
data_vec = data_vec_tmp.groupby("user_id").mean()

data_vec.to_csv(data_vec_filepath, header=False, index=True)
