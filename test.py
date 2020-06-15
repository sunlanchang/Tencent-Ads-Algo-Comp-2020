# %%
import random
import unittest

from transformers import is_torch_available

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pandas as pd

if is_torch_available():
    from transformers import (
        BertConfig,
        BertModel,
        BertForMaskedLM,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertForMultipleChoice,
    )
    from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST


# %%
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
EPOCHS = 10


# %%
creative_id_seq = []
cnt = 0
with open('word2vec/userid_creative_ids.txt', 'r') as f:
    for text in f:
        creative_id_seq.append(text.strip())
        cnt += 1
        if cnt == 90:
            break
with open('tmp/tmp.txt', 'w')as f:
    f.write('[PAD]\n[UNK]\n[CLS]\n[SEP]\n')
    s = set()
    for seq in creative_id_seq:
        seq = seq.split(' ')
        s = s | set(seq)
    for e in s:
        f.write(str(e)+'\n')


# %%
user_train = pd.read_csv(
    'data/train_preliminary/user.csv').sort_values(['user_id'], ascending=(True,))
Y_gender = user_train['gender'].values
Y_age = user_train['age'].values
Y_gender = Y_gender - 1
Y_age = Y_age - 1
# Y_age = to_categorical(Y_age)


# %%
tokenizer = BertTokenizer('tmp/tmp.txt')
print(tokenizer.get_vocab())
sample_txt = '456 1 23 456 89 89'
# tokenizer.tokenize(sample_txt)


# %%

encoding = tokenizer.encode_plus(
    sample_txt,
    max_length=32,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',  # Return PyTorch tensors
)
# encoding.keys()
# encoding['input_ids']
# encoding['attention_mask']
# tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
