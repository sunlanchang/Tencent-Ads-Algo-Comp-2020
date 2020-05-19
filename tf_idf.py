# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
# %%
user = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
# %%
Y_train_gender = user.gender
Y_train_age = user.age
corpus = []
f = open('word2vec/userid_creativeids.txt', 'r')
flag = 0
for row in f:
    # row = [[int(e) for e in seq] for seq in row.strip().split(' ')]
    row = row.strip()
    corpus.append(row)
    flag += 1
    if flag == 100:
        break
# %%
Y_train_gender = Y_train_gender.iloc[:flag]-1
Y_train_age = Y_train_age.iloc[:flag]-1
# %%
vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    min_df=1,
    # max_features=128,
    dtype=np.float32,
)
X_train = vectorizer.fit_transform(corpus)
print(X_train.shape)
# %%
X_train_gender, X_val_gender, Y_train_gender, Y_val_gender = train_test_split(
    X_train, Y_train_gender, train_size=0.9, random_state=1)
lgb_train_gender = lgb.Dataset(X_train_gender, Y_train_gender)
lgb_eval_gender = lgb.Dataset(
    X_val_gender, Y_val_gender, reference=lgb_train_gender)

X_train_age, X_val_age, Y_train_age, Y_val_age = train_test_split(
    X_train, Y_train_age, train_size=0.9, random_state=1)
lgb_train_age = lgb.Dataset(X_train_age, Y_train_age)
lgb_eval_age = lgb.Dataset(
    X_val_age, Y_val_age, reference=lgb_train_age)
# %%


def LGBM_gender(epoch, early_stopping_rounds):
    params_gender = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'binary_error'},  # evaluate指标
        'max_depth': -1,             # 不限制树深度
        # 更高的accuracy
        'max_bin': 2**10-1,

        'num_leaves': 2**10,
        'min_data_in_leaf': 1,
        'learning_rate': 0.01,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        # 'is_provide_training_metric': True,
        'verbose': 1
    }
    print('Start training...')
    # train
    gbm = lgb.train(params_gender,
                    lgb_train_gender,
                    num_boost_round=epoch,
                    valid_sets=lgb_eval_gender,
                    # early_stopping_rounds=5,
                    )
    print('training done!')
    print('Saving model...')
    # save model to file
    gbm.save_model('tmp/model_gender.txt')
    print('save model done!')
    return gbm
# %%


def LGBM_age(epoch, early_stopping_rounds):
    params_age = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        "num_class": 10,
        # fine-tuning最重要的三个参数
        'num_leaves': 2**10-1,
        'max_depth': -1,             # 不限制树深度
        'min_data_in_leaf': 1,
        # 更高的accuracy
        # 'max_bin': 2**9-1,
        # 'num_iterations': 50,  # epoch
        'metric': {'multi_logloss', 'multi_error'},
        'learning_rate': 0.01,

        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'verbose': 1
    }
    print('Start training...')
    # train
    gbm = lgb.train(params_age,
                    lgb_train_age,
                    num_boost_round=epoch,
                    valid_sets=lgb_eval_age,
                    early_stopping_rounds=1000,
                    )
    print('Saving model...')
    # save model to file
    gbm.save_model('tmp/model_age.txt')
    print('save model done!')
    return gbm


LGBM_age(epoch=10, early_stopping_rounds=1000)
# %%


LGBM_gender(epoch=50, early_stopping_rounds=1000)

# %%
