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

X_val = df_val[feature_columns]
y_val_gender = df_val[label_gender]
y_val_gender.gender = y_val_gender.gender-1

X_test = df_test[feature_columns]

print('Loading data uses {:.1f}s'.format(time.time()-start))
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train_gender)
lgb_eval = lgb.Dataset(X_val, y_val_gender, reference=lgb_train)


# %%
# specify your configurations as a dict
def ageLGBM():
    params_age = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }


# %%
def genderLGBM():
    params_gender = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'num_class': 2,            # 多分类类别
        'metric': 'binary_logloss',  # evaluate指标
        'max_depth': -1,             # 不限制树深度
        'num_leaves': 31,
        'min_data_in_leaf': 1,
        'learning_rate': 0.1,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'is_provide_training_metric': True,
        'verbose': 1
    }
    print('Starting training...')
    # train
    gbm = lgb.train(params_gender,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    # early_stopping_rounds=5,
                    )
    print('training done!')
    print('Saving model...')
    # save model to file
    gbm.save_model('tmp/model.txt')
    print('save model done!')
    return gbm


# %%
gbm = genderLGBM()
# %%
if __name__ == "__main__":
    print('Starting predicting...')
    y_pred_probability = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    threshold = 0.5
    y_pred = np.where(y_pred_probability > threshold, 1, 0)
    # eval
    print('threshold: {:.1f} The accuracy of prediction is:{:.2f}'.format(threshold,
                                                                          accuracy_score(y_val_gender, y_pred)))


# %%
