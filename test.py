from multiprocessing import Process
import multiprocessing
import pandas as pd
# 子进程要执行的代码
from multiprocessing import Pool, cpu_count
import os
import time

# print(multiprocessing.cpu_count())
df_creativeid_embedding = pd.read_hdf(
    'word2vec/df_creativeid_embedding.h5',
    key='df_creativeid_embedding', mode='r')

with open('word2vec/userid_creativeids.txt', 'r')as f:
    seq_creative_id = f.readlines()
seq_creative_id = [[str(e) for e in line.strip().split(' ')]
                   for line in seq_creative_id]


def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    print(df_creativeid_embedding.iloc[0, 0])


if __name__ == '__main__':
    p = Pool(4)
    indexes = [(0, 1000000), (1000000, 1900000)]
    for index in indexes:
        p.apply_async(long_time_task, args=(index,))
    print('等待所有子进程完成。')
    p.close()
    p.join()


# for user in tqdm.tqdm(range(len(seq_creative_id))):
#     user_em = df_creativeid_embedding.loc[seq_creative_id[user]].mean()
