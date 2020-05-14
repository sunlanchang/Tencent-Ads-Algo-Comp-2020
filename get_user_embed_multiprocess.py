from multiprocessing import Process
import multiprocessing
import pandas as pd
# 子进程要执行的代码
from multiprocessing import Pool, cpu_count
import os
import time
import tqdm
import pickle
# print(multiprocessing.cpu_count())


def long_time_task(i, start, end):
    pid = os.getpid()
    columns = ['c'+str(i) for i in range(128)]
    data = {}
    for col_name in columns:
        data[col_name] = pd.Series([], dtype='float')
    df_user_embedding = pd.DataFrame(data)

    for idx in range(start, end):
        user_emb = df_creativeid_embedding.loc[seq_creative_id[idx]].mean()
        df_user_embedding = df_user_embedding.append(
            user_emb, ignore_index=True)

        if idx != start and (idx-start) % 500 == 0:
            print('进程{}: {}/{}'.format(pid, idx-start, end-start))
        if idx != start and (idx-start) % 5000 == 0:
            df_user_embedding.to_hdf(
                '/tmp/df_user_embedding{}.h5'.format(i), key='df_user_embedding{}'.format(i), mode='w')
            # break


if __name__ == '__main__':
    df_creativeid_embedding = pd.read_hdf(
        'word2vec/df_creativeid_embedding.h5',
        key='df_creativeid_embedding', mode='r')

    # with open('word2vec/userid_creativeids.txt', 'r')as f:
    #     seq_creative_id = f.readlines()
    # seq_creative_id = [[str(e) for e in line.strip().split(' ')]
    #                    for line in seq_creative_id]

    # with open('word2vec/seq_creative_id.pkl', 'wb') as f:
    #     pickle.dump(seq_creative_id, f)
    #     print('pickle done.')
    with open('word2vec/seq_creative_id.pkl', 'rb') as f:
        print('start reading...')
        seq_creative_id = pickle.load(f)
        print('read pickle done.')

    print('当前母进程: {}'.format(os.getpid()))
    p = Pool(os.cpu_count())

    my_cpu_count = os.cpu_count()
    num_user = 1900000
    unit = num_user//my_cpu_count
    indexes = []
    for idx in range(my_cpu_count):
        indexes.append((unit*idx, unit*(idx+1)))
    if unit*(idx+1) != num_user:
        indexes.append((unit*(idx+1), num_user))
    import time
    time_start = time.time()
    for i, (start, end) in enumerate(indexes):
        p.apply_async(long_time_task, args=(i, start, end))
    p.close()
    p.join()
    print('等待所有子进程完成。')
    print('共使用 {:.2f} min.'.format((time.time()-time_start)/60))

    # for user in tqdm.tqdm(range(len(seq_creative_id))):
    #     user_em = df_creativeid_embedding.loc[seq_creative_id[user]].mean()
