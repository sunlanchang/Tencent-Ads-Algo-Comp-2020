# %%
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.layers import LSTM
# import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import multi_gpu_model
import gc
from mail import mail
# %%
samples = 900000
columns = ['c'+str(n) for n in range(128)]
train_data = pd.read_csv('word2vec/creative_id.csv',
                         nrows=samples, skiprows=None)
X_train = train_data[columns].values
del train_data
# %%
test_data = pd.read_csv('word2vec/creative_id.csv',
                        names=columns,
                        skiprows=900001)
gc.collect()
X_test = test_data[columns].values
del test_data
gc.collect()
# %%
user_train = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_gender = user_train['gender'].values
Y_age = user_train['age'].values
# %%
user_id_test = pd.read_csv(
    'data/test/clicklog_ad_user_test.csv').sort_values(['user_id'], ascending=(True,)).user_id.unique()
ans = pd.DataFrame({'user_id': user_id_test})
# %%
# %%


def gender_lstm():
    X_train = X_train.reshape(-1, 128, 1)
    Y_gender = Y_gender[:samples] - 1
    model = keras.Sequential(
        [
            LSTM(1024),
            Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_gender, validation_split=0.1,
              epochs=150, batch_size=1024)
    mail('train lstm done!')
# %%


def create_gender_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(128,)),
            layers.Dense(256, activation="relu"),
            # layers.Dense(512, activation="relu"),
            layers.Dense(1024, activation="relu"),
            # layers.Dense(512, activation="relu"),
            # layers.Dense(256, activation="relu"),
            layers.Dense(128, activation='relu'),
            # layers.Dense(2, activation='softmax', name='classifier')
            Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy',
                  #   optimizer='sgd', metrics=['accuracy'])
                  optimizer='adam', metrics=['accuracy'])

    # model.summary()
    return model

# %%


def create_age_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(128,)),
            layers.Dense(256, activation="relu"),
            # layers.Dense(512, activation="relu"),
            # layers.Dense(1024, activation="relu"),
            # layers.Dense(512, activation="relu"),
            # layers.Dense(256, activation="relu"),
            layers.Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ]
    )
    model.compile(loss='categorical_crossentropy',
                  #   optimizer='rmsprop', metrics=['accuracy'])
                  optimizer='adam', metrics=['accuracy'])

    # model.summary()
    return model

# %%


def train_gender(X, Y, X_test, train=True, epoch=10, batch_size=1024):
    # 类别转换为0和1
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y_encoded = encoder.transform(Y)
    if train:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        model = create_gender_model()
        model.fit(X, Y_encoded, validation_split=0.1,
                  batch_size=batch_size, epochs=epoch)

        X_test = scaler.transform(X_test)
        y_pre = model.predict(X_test)
        threshold = 0.5
        y_pred_gender = np.where(y_pre > threshold, 1, 0)
        return y_pred_gender
    else:
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(
            build_fn=create_gender_model, epochs=epoch, batch_size=batch_size, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        results = cross_val_score(pipeline, X, Y_encoded, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" %
              (results.mean()*100, results.std()*100))


y_gender = train_gender(X_train, Y_gender, X_test,
                        train=True, epoch=300, batch_size=4096)
# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model = model
# parallel_model.fit(X, Y, epochs=10, batch_size=batch_size)
# gender_pred = parallel_model.predict(X_test, batch_size=batch_size)
# return gender_pred
# %%


def train_age(X, Y, X_test, train=True, epoch=10, batch_size=1024):
        # 类别转换为0和1
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    if train:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        Y = to_categorical(Y)
        model = create_age_model()
        model.fit(X, Y, validation_split=0.1,
                  batch_size=batch_size, epochs=epoch)

        X_test = scaler.transform(X_test)
        y_pre = model.predict(X_test)
        y_pred_age = np.argmax(y_pre, axis=1)

        return y_pred_age
    else:
        # estimator = KerasClassifier(
        #     build_fn=create_gender_model, epochs=epoch, batch_size=batch_size, verbose=0)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(
            build_fn=create_age_model, epochs=epoch, batch_size=batch_size, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, Y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" %
              (results.mean()*100, results.std()*100))


# %%
y_gender = train_gender(X_train, Y_gender, X_test,
                        train=True, epoch=300, batch_size=4096)
# %%
y_age = train_age(X_train, Y_age, X_test,
                  train=True, epoch=500, batch_size=4096)
# %%
ans['predicted_age'] = y_age+1
ans['predicted_gender'] = y_gender+1
# %%
ans.to_csv('data/ans/word2vec.csv',
           columns=['user_id', 'predicted_age', 'predicted_gender'],
           header=True,
           index=False,
           )

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
