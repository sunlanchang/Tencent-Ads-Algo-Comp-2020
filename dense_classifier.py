# %%
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
# 合成用户embedding
path = "word2vec/wordvectors.kv"
wv = KeyedVectors.load(path, mmap='r')
# %%


# %%
# 将同一个用户creative_id相加平均后即为一个用户的Embedding
columns = ['c'+str(i) for i in range(128)]


# %%
