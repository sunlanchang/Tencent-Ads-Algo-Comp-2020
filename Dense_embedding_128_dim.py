# %%
import gc
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import layers
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
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)
# import keras
# %%
samples = 100000
columns = ['c'+str(n) for n in range(128)]
train_data = pd.read_csv('word2vec/creative_id.csv',
                         #  nrows=900000, skiprows=None)
                         nrows=samples, skiprows=None)
X_train = train_data[columns].values
del train_data
gc.collect()
# %%
test_data = pd.read_csv('word2vec/creative_id.csv',
                        names=columns,
                        # skiprows=900001)
                        skiprows=samples)

X_test = test_data[columns].values
del test_data
gc.collect()
# %%
# 得到标签
user_train = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_gender = user_train['gender'].values
Y_age = user_train['age'].values
# %%
user_id_test = pd.read_csv(
    'data/test/clicklog_ad_user_test.csv').sort_values(['user_id'], ascending=(True,)).user_id.unique()
ans = pd.DataFrame({'user_id': user_id_test})
# %%
# create the model
X_train = X_train.reshape(-1, 128, 1)
Y_gender = Y_gender[:samples] - 1
model = keras.Sequential(
    [
        LSTM(100),
        Dense(1, activation='sigmoid')
    ]
)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_gender, validation_split=0.1, epochs=3, batch_size=64)
# %%
inputs = tf.random.normal([32, 128, 1])
output = model.predict(inputs)
output.shape
# %%
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=3, batch_size=64)
# %%


def gender_dense():
    model = keras.Sequential(
        [
            keras.Input(shape=(128,)),
            layers.Dense(256, activation="elu"),
            layers.Dense(512, activation="elu"),
            # layers.Dense(1024, activation="elu"),
            # layers.Dense(512, activation="elu"),
            layers.Dense(256, activation="elu"),
            layers.Dense(128, activation='elu'),
            # layers.Dense(2, activation='softmax', name='classifier')
            Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    # model.summary()
    return model

# %%


def age_dense():
    model = keras.Sequential(
        [
            keras.Input(shape=(128,)),
            layers.Dense(256, activation="elu"),
            layers.Dense(512, activation="elu"),
            # layers.Dense(1024, activation="elu"),
            # layers.Dense(512, activation="elu"),
            layers.Dense(256, activation="elu"),
            layers.Dense(128, activation='elu'),
            Dense(10, activation='softmax')
        ]
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
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
        model = gender_dense()
        model.fit(X, Y_encoded, batch_size=batch_size, epochs=epoch)

        X_test = scaler.transform(X_test)
        y_pre = model.predict(X_test)
        threshold = 0.5
        y_pred_gender = np.where(y_pre > threshold, 1, 0)
        return y_pred_gender
    else:
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(
            build_fn=gender_dense, epochs=epoch, batch_size=batch_size, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        results = cross_val_score(pipeline, X, Y_encoded, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" %
              (results.mean()*100, results.std()*100))

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
        model = age_dense()
        model.fit(X, Y, batch_size=batch_size, epochs=epoch)

        X_test = scaler.transform(X_test)
        y_pre = model.predict(X_test)
        y_pred_age = np.argmax(y_pre, axis=1)

        return y_pred_age
    else:
        # estimator = KerasClassifier(
        #     build_fn=gender_dense, epochs=epoch, batch_size=batch_size, verbose=0)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(
            build_fn=age_dense, epochs=epoch, batch_size=batch_size, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, Y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" %
              (results.mean()*100, results.std()*100))


# %%
y_gender = train_gender(X_train, Y_gender, X_test,
                        train=False, epoch=50, batch_size=4096)
y_age = train_age(X_train, Y_age, X_test,
                  train=False, epoch=50, batch_size=4096)
# %%
ans['predicted_age'] = y_age+1
ans['predicted_gender'] = y_gender+1
# %%
ans.to_csv('data/ans/word2vec.csv',
           columns=['user_id', 'predicted_age', 'predicted_gender'],
           header=True,
           index=False,
           )
