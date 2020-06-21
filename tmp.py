import pandas as pd
import numpy as np

DATA = {}
DATA['ad_id_emb'] = np.load(
    'tmp/embeddings_0.npy', allow_pickle=True)
arr = DATA['ad_id_emb']

result = []
for i in range(arr.shape[-1]):
    result.append([np.mean(arr[:, i]), np.std(arr[:, i])])
dfi = pd.DataFrame(result, columns=['mean', 'std'])
print(dfi.describe().T)
# from gensim.models import Word2Vec
# from gensim.models.callbacks import CallbackAny2Vec


# class LossCallback(CallbackAny2Vec):
#     '''Callback to print loss after each epoch.'''

#     def __init__(self):
#         self.epoch = 0
#         self.loss_to_be_subed = 0

#     def on_epoch_end(self, model):
#         loss = model.get_latest_training_loss()
#         loss_now = loss - self.loss_to_be_subed
#         self.loss_to_be_subed = loss
#         print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
#         self.epoch += 1


# model = Word2Vec(common_texts, size=100, window=5, min_count=1,
#                  compute_loss=True, callbacks=[LossCallback()])


# tmp = df.groupby(sentence_id,
#                  as_index=False)[word_id].agg({list_col_nm: list})
# sentences = tmp[list_col_nm].values.tolist()
# all_words_vocabulary = df[word_id].unique().tolist()
# del tmp[list_col_nm]
# gc.collect()

# if embedding_type == 'w2v':
#     model = Word2Vec(
#         sentences,
#         size=emb_size,
#         window=150,
#         workers=n_jobs,
#         min_count=1,  # 最低词频. min_count>1会出现OOV
#         sg=sg,  # 1 for skip-gram; otherwise CBOW.
#         hs=hs,  # If 1, hierarchical softmax will be used for model training
#         negative=negative,  # hs=1 + negative 负采样
#         iter=epoch,
#         seed=0)
