# %%
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import time
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.utils import multi_gpu_model
import gc
# %%
samples = 1000
all_train_data = pd.read_csv('word2vec/data_vec_product_category_industry_16dimension.csv',
                             nrows=900000, skiprows=None)
#  nrows=samples, skiprows=None).sort_values(['user_id'], ascending=(True,))
columns = all_train_data.columns.values.tolist()
test_data = pd.read_csv('word2vec/data_vec_product_category_industry_16dimension.csv',
                        names=columns,
                        skiprows=900001,
                        # nrows=samples,
                        ).sort_values(['user_id'], ascending=(True,))

user_train = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_gender = user_train['gender'].values
Y_age = user_train['age'].values

all_train_data['gender'] = user_train.gender
all_train_data['age'] = user_train.age

TRAIN_DATA_PERCENT = 0.9
mask = np.random.rand(len(all_train_data)) < TRAIN_DATA_PERCENT
df_train = all_train_data[mask]
df_val = all_train_data[~mask]

X_train = df_train[columns].values
Y_train_gender = df_train.gender.values
Y_train_age = df_train.age.values

X_val = df_val[columns].values
Y_val_gender = df_val.gender.values
Y_val_age = df_val.age.values
# del train_data
# gc.collect()
X_test = test_data[columns].values
# del test_data
# gc.collect()

user_id_test = pd.read_csv(
    'data/test/clicklog_ad_user_test.csv').sort_values(['user_id'], ascending=(True,)).user_id.unique()
ans = pd.DataFrame({'user_id': user_id_test})
# %%
# 构建性别数据
encoder = LabelEncoder()
encoder.fit(Y_train_gender)
Y_train_gender = encoder.transform(Y_train_gender)
Y_val_gender = encoder.transform(Y_val_gender)

lgb_train_gender = lgb.Dataset(X_train, Y_train_gender)
lgb_eval_gender = lgb.Dataset(X_val, Y_val_gender, reference=lgb_train_gender)
# 构建年龄数据
encoder = LabelEncoder()
encoder.fit(Y_train_age)
Y_train_age = encoder.transform(Y_train_age)
Y_val_age = encoder.transform(Y_val_age)

lgb_train_age = lgb.Dataset(X_train, Y_train_age)
lgb_eval_age = lgb.Dataset(X_val, Y_val_age, reference=lgb_train_age)

# %%


def LGBM_gender():
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
                    num_boost_round=50,
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


def LGBM_age():
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

        'metric': {'multi_logloss', 'multi_error'},
        'learning_rate': 0.1,

        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'verbose': 1
    }
    print('Start training...')
    # train
    gbm = lgb.train(params_age,
                    lgb_train_age,
                    num_boost_round=50,
                    valid_sets=lgb_eval_age,
                    # early_stopping_rounds=5,
                    )
    print('Saving model...')
    # save model to file
    gbm.save_model('tmp/model_age.txt')
    print('save model done!')
    return gbm


# %%
gbm_gender = LGBM_gender()
# %%
gbm_age = LGBM_age()
# %%


def evaluate():
    print('Start predicting...')
    y_pred_gender_probability = gbm_gender.predict(
        X_val, num_iteration=gbm_gender.best_iteration)
    threshold = 0.5
    y_pred_gender = np.where(y_pred_gender_probability > threshold, 1, 0)
    # eval
    print('threshold: {:.1f} The accuracy of prediction is:{:.2f}'.format(threshold,
                                                                          accuracy_score(Y_val_gender, y_pred_gender)))
    print('Start evaluate data predicting...')
    y_pred_age_probability = gbm_age.predict(
        X_val, num_iteration=gbm_age.best_iteration)
    y_pred_age = np.argmax(y_pred_age_probability, axis=1)
    # eval
    print('The accuracy of prediction is:{:.2f}'.format(
        accuracy_score(Y_val_age, y_pred_age)))

    # d = {'user_id': X_val.user_id.values.tolist(), 'gender': Y_pred_gender.tolist(),
    #      'age': y_pred_age.tolist()}
    # ans_df = pd.DataFrame(data=d)
    # # 投票的方式决定gender、age
    # ans_df_grouped = ans_df.groupby(['user_id']).agg(
    #     lambda x: x.value_counts().index[0])
    # ans_df_grouped.gender = ans_df_grouped.gender+1
    # ans_df_grouped.age = ans_df_grouped.age+1
    # ans_df_grouped.to_csv('data/ans.csv', header=True)


# %%
evaluate()
# %%


def test():
    print('Start predicting test gender data ...')
    y_pred_gender_probability = gbm_gender.predict(
        X_test, num_iteration=gbm_gender.best_iteration)
    threshold = 0.5
    y_pred_gender = np.where(y_pred_gender_probability > threshold, 1, 0)

    print('Start predicting test age data ...')
    y_pred_age_probability = gbm_age.predict(
        X_test, num_iteration=gbm_age.best_iteration)
    y_pred_age = np.argmax(y_pred_age_probability, axis=1)

    ans['predicted_age'] = y_pred_age+1
    ans['predicted_gender'] = y_pred_gender+1
    ans.to_csv('data/ans/LGBM.csv', header=True, index=False,
               columns=['user_id', 'predicted_age', 'predicted_gender'])

    # ans_df = pd.DataFrame(data=d)
    # 投票的方式决定gender、age
    # ans_df_grouped = ans_df.groupby(['user_id']).agg(
    #     lambda x: x.value_counts().index[0])
    # ans_df_grouped['user_id'] = ans_df_grouped.index
    # ans_df_grouped.gender = ans_df_grouped.gender+1
    # ans_df_grouped.age = ans_df_grouped.age+1
    # columns_order = ['user_id', 'predicted_age', 'predicted_gender']
    # ans_df_grouped[columns_order].to_csv(
    #     'data/ans_test.csv', header=True, columns=['user_id', 'predicted_age', 'predicted_gender'], index=False)
    # print('Done!!!')


test()
# %%
# %%
# df_train = df_train.sort_values(
#     ["user_id"], ascending=(True,))

# # %%


# def get_batch(file_name,):
#     for row in open(file_name, "r"):
#         yield 1


# for line in get_batch('data/train_data.csv'):
# for line in get_batch('test.py'):
# print(line)
# break
# %%
# 合成用户embedding
# path = "word2vec/wordvectors.kv"
# wv = KeyedVectors.load(path, mmap='r')
# with open('word2vec/userid_creativeids.txt', 'r')as f:
#     lines = f.readlines()
# lines = [[int(e) for e in line.split(' ')] for line in lines]
# number_train_user = 900000
# number_test_user = 1000000
# user_train = lines[:number_train_user]
# user_test = lines[number_train_user:]
# columns = ['c'+str(i) for i in range(128)]
# data = {}
# for col_name in columns:
#     data[col_name] = pd.Series([], dtype='float')
# df_user_train = pd.DataFrame(data)
# df_user_test = pd.DataFrame(data)
# # %%
# for line in tqdm.tqdm(user_train):
#     user_embedding_train = np.zeros(128)
#     for creative_id in line:
#         user_embedding_train += wv[str(creative_id)]
#     user_embedding_train = user_embedding_train / len(line)
#     tmp = pd.DataFrame(user_embedding_train.reshape(-1,
#                                                     len(user_embedding_train)), columns=columns)
#     df_user_train = df_user_train.append(tmp)
# # %%
# for line in tqdm.tqdm(user_test):
#     user_embedding_test = np.zeros(128)
#     for creative_id in line:
#         user_embedding_test += wv[str(creative_id)]
#     user_embedding_test = user_embedding_test / len(line)
#     tmp = pd.DataFrame(user_embedding_test.reshape(-1,
#                                                    len(user_embedding_train)), columns=columns)
#     df_user_test = df_user_test.append(tmp)
# # %%
# # 将同一个用户creative_id相加平均后即为一个用户的Embedding
# all_train_data = pd.read_csv(
#     'data/train_preliminary/clicklog_ad_user_train_eval_test.csv')
# all_train_data = all_train_data.sort_values(
#     ["user_id"], ascending=(True))
# # %%
# all_test_data = pd.read_csv(
#     'data/test/clicklog_ad_user_test.csv')
# all_test_data = all_test_data.sort_values(
#     ["user_id"], ascending=(True))
# # %%
# assert df_user_train.shape[0] == all_train_data.shape[0]
# df_user_train['user_id'] = all_train_data['user_id']
# df_user_train['gender'] = all_train_data['gender']
# df_user_train['age'] = all_train_data['age']
# df_user_train.to_hdf('word2vec/df_user_train_test.h5',
#                      key='df_user_train', mode='w')
# # %%
# assert df_user_test.shape[0] == all_test_data.shape[0]
# df_user_test['user_id'] = all_test_data['user_id']
# df_user_test.to_hdf('word2vec/df_user_train_test.h5',
#                     key='df_user_test', mode='a')


# %%
