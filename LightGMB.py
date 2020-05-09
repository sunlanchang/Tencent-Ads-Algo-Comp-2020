# %%
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
# %%
print('Loading all data...')
start = time.time()
all_train_data = pd.read_csv('data/train_preliminary/clicklog_ad_user.csv')
df_test = pd.read_csv('data/test/clicklog_ad.csv')
print('Split data into train and validation...')
TRAIN_DATA_PERCENT = 0.9
msk = np.random.rand(len(all_train_data)) < TRAIN_DATA_PERCENT
df_train = all_train_data[msk]
df_val = all_train_data[~msk]
feature_columns = df_train.columns.values.tolist()
feature_columns.remove('age')
feature_columns.remove('gender')
label_age, label_gender = ['age'], ['gender']

X_train = df_train[feature_columns]
y_train_gender = df_train[label_gender]
# set label 0 and 1
y_train_gender.gender = y_train_gender.gender-1

y_train_age = df_train[label_age]
y_train_age.age = y_train_age.age-1

X_val = df_val[feature_columns]
y_val_gender = df_val[label_gender]
y_val_gender.gender = y_val_gender.gender-1

y_val_age = df_val[label_age]
y_val_age.age = y_val_age.age-1


X_test = df_test[feature_columns]

print('Loading data uses {:.1f}s'.format(time.time()-start))
categorical_feature = ['industry', 'advertiser_id',
                       'product_category', 'product_id', 'ad_id', 'creative_id', 'user_id']
# 构建性别数据
lgb_train_gender = lgb.Dataset(
    X_train, y_train_gender, feature_name=feature_columns, categorical_feature=categorical_feature)
lgb_eval_gender = lgb.Dataset(
    X_val, y_val_gender, reference=lgb_train_gender, feature_name=feature_columns, categorical_feature=categorical_feature)
# 构建年龄数据
lgb_train_age = lgb.Dataset(
    X_train, y_train_age, feature_name=feature_columns, categorical_feature=categorical_feature)
lgb_eval_age = lgb.Dataset(
    X_val, y_val_age, reference=lgb_train_age, feature_name=feature_columns, categorical_feature=categorical_feature)


# %%
def LGBM_gender():
    params_gender = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'binary_error'},  # evaluate指标
        'max_depth': -1,             # 不限制树深度
        'num_leaves': 2**8,
        'min_data_in_leaf': 1,
        'learning_rate': 0.1,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        # 'is_provide_training_metric': True,
        'verbose': 1
    }
    print('Starting training...')
    # train
    gbm = lgb.train(params_gender,
                    lgb_train_gender,
                    num_boost_round=100,
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
        'num_leaves': 2**8-1,
        'max_depth': -1,             # 不限制树深度
        'min_data_in_leaf': 1,

        'metric': {'multi_logloss', 'multi_error'},
        'learning_rate': 0.1,

        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'verbose': 1
    }
    print('Starting training...')
    # train
    gbm = lgb.train(params_age,
                    lgb_train_age,
                    num_boost_round=50,
                    valid_sets=lgb_eval_age,
                    # early_stopping_rounds=5,
                    )
    print('training done!')
    print('Saving model...')
    # save model to file
    gbm.save_model('tmp/model_age.txt')
    print('save model done!')
    return gbm


# %%
# gbm_gender = LGBM_gender()

# %%
# gbm_gender = LGBM_gender()
# gbm_age = LGBM_age()
gbm_gender = lgb.Booster(model_file='tmp/model_gender.txt')
gbm_age = lgb.Booster(model_file='tmp/model_age.txt')

# %%
# if __name__ == "__main__":
print('Starting predicting...')
y_pred_gender_probability = gbm_gender.predict(
    X_val, num_iteration=gbm_gender.best_iteration)
threshold = 0.5
y_pred_gender = np.where(y_pred_gender_probability > threshold, 1, 0)
# eval
print('threshold: {:.1f} The accuracy of prediction is:{:.2f}'.format(threshold,
                                                                      accuracy_score(y_val_gender, y_pred_gender)))
# %%
print('Starting predicting...')
y_pred_age_probability = gbm_age.predict(
    X_val, num_iteration=gbm_age.best_iteration)
threshold = 0.5
y_pred_age = np.argmax(y_pred_age_probability, axis=1)
# eval
print('The accuracy of prediction is:{:.2f}'.format(
    accuracy_score(y_val_age, y_pred_age)))
# %%
d = {'user_id': X_val.user_id.values.tolist(), 'gender': y_pred_gender.tolist(),
     'age': y_pred_age.tolist()}
ans_df = pd.DataFrame(data=d)
# 投票的方式决定gender、age
ans_df_grouped = ans_df.groupby(['user_id']).agg(
    lambda x: x.value_counts().index[0])
ans_df_grouped.gender = ans_df_grouped.gender+1
ans_df_grouped.age = ans_df_grouped.age+1
ans_df_grouped.to_csv('data/ans.csv', header=True, index=False)
# userid = np.hstack((np.array(X_val.user_id).reshape(
#     (X_val.user_id.shape[0], 1)), y_pred_gender))
# userid_gender = np.hstack((userid, y_pred_gender))

# %%
# for leaves in range(10, 13):
#     gbm_age = LGBM_age(leaves)
#     y_pred_probability = gbm_age.predict(
#         X_val, num_iteration=gbm_age.best_iteration)
#     y_pred = np.argmax(y_pred_probability, axis=1)
#     print('v'*20)
#     print('leaves: ', leaves)
#     print('The accuracy of prediction is:{:.2f}'.format(
#         accuracy_score(y_val_age, y_pred)))
#     print('v'*20)


# %%
