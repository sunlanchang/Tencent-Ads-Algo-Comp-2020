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
tokenizer.tokenize(sample_txt)


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
encoding.keys()
encoding['input_ids']
encoding['attention_mask']
tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])


# %%


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes=10):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        # self.bert = model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
# %%


class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# %%


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


# tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier()

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


# %%


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        print(outputs.shape)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# %%
ds = GPReviewDataset(creative_id_seq[:90], Y_age[:90], tokenizer, 100)
dataloader = DataLoader(ds, batch_size=1)


# %%
# train_epoch(model, dataloader, loss_fn, optimizer,
#             device, scheduler, len(dataloader))

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc
