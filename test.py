# %%
import pandas as pd
import os
# %%
user_embeddings = []
for i in range(os.cpu_count()):
    tmp = pd.read_hdf(
        '/tmp/df_user_embedding{}.h5'.format(i), key='df_user_embedding{}'.format(i), mode='r')
    user_embeddings.append(tmp)


# %%
