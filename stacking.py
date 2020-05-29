# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from mail import mail
# %%
user = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_train_gender = user.gender
Y_train_age = user.age
corpus = []
f = open('word2vec/userid_creativeids.txt', 'r')
# train_examples = 100
# test_examples = 200
# train_test = 300
train_test = 1900000
train_examples = 900000
test_examples = 1000000
flag = 0
for row in f:
    # row = [[int(e) for e in seq] for seq in row.strip().split(' ')]
    row = row.strip()
    corpus.append(row)
    flag += 1
    if flag == train_test:
        break
# %%
Y_train_gender = Y_train_gender.iloc[:train_examples]-1
Y_train_age = Y_train_age.iloc[:train_examples]-1
# %%
min_df = 30
max_df = 0.001
vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    min_df=min_df,
    # max_df=max_df,
    # max_features=128,
    dtype=np.float32,
)
all_data = vectorizer.fit_transform(corpus)
print('(examples, features)', all_data.shape)
print('train tfidf done! min_df={}, max_df={} shape is {}'.format(
    min_df, max_df, all_data.shape[1]))
mail('train tfidf done! min_df={}, max_df={} shape is {}'.format(
    min_df, max_df, all_data.shape[1]))
# %%
train_val = all_data[:train_examples, :]
# %%
X_test = all_data[train_examples:(train_examples+test_examples), :]
# %%
test_user_id = pd.read_csv(
    'data/test/click_log.csv').sort_values(['user_id'], ascending=(True)).user_id.unique()
# %%
test_user_id = test_user_id[:test_examples]
