# %%
import pandas as pd
import numpy as np
# %%
all_train_data = pd.read_csv('data/train_preliminary/clicklog_ad_user.csv')
df_test = pd.read_csv('data/test/clicklog_ad.csv')


# %%
all_train_data.to_hdf('data/clicklog_ad_user.h5',
                      key='all_train_data', mode='w')
df_test.to_hdf('data/clicklog_ad_user.h5', key='df_test', mode='a')


# %%
all_train_data.read_hdf('data/clicklog_ad_user.h5',
                        key='all_train_data', mode='r')
df_test.read_hdf('data/clicklog_ad_user.h5', key='df_test', mode='r')

# %%
all_train_data_sorted = all_train_data.sort_values(
    ["user_id", "time"], ascending=(True, True))
userid_creative_ids = all_train_data_sorted.groupby(
    'user_id')['time'].apply(list).reset_index(name='creative_ids')
# %%
with open('data/userid_creativeids.txt', 'w')as f:
    for ids in userid_creative_ids.creative_ids:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
# %%
