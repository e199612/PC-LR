import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('UB_clear_pv.csv').drop(['Unnamed: 0'], axis=1)

date_list = sorted(list(set(data['date'])))
train_date = date_list[:4]

# create training data
train_b = []
for date in train_date:
    print(date)
    train_data = data[data['date'] == date]
    train_uid_list = list(set(train_data['uid']))

    for user in tqdm(train_uid_list):
        tmp = train_data[train_data['uid'] == user]
        tmp = tmp.sort_values(by='timestamp')

        item = list(tmp['iid']) # click stream item
        cat = list(tmp['cid']) # click stream item

        # add the {user:click seq}
        train_b.append((user,item,cat))

with open('ali_user_seq_4days.pkl','wb') as f:
    pkl.dump(train_b,f)