import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('UB_clear_pv.csv').drop(['Unnamed: 0'], axis=1)
date_list = sorted(list(set(data['date'])))

# use one day as training data and last days as testing data
train_date = date_list[0]
test_date = date_list[1:]

# training data
train_data = data[data['date'] == train_date]
train_uid_list = list(set(train_data['uid']))

train_b = []
for user in tqdm(train_uid_list):
    tmp = train_data[train_data['uid'] == user]
    tmp = tmp.sort_values(by='timestamp')
    
    item = list(tmp['iid']) # click stream item
        
    # add the {user:click seq}
    train_b.append((user,item))

with open('split_data/ali_user_seq_11_25.pkl','wb') as f:
    pkl.dump(train_b,f)

# testing data
for dat in test_date:
    print(dat)
    test_data = data[data['date'] == dat]
    test_uid_list = list(set(test_data['uid']))
    test_b = []
    for user in tqdm(test_uid_list):
        tmp = test_data[test_data['uid'] == user]
        tmp = tmp.sort_values(by='timestamp')

        item = list(tmp['iid']) # click stream item

        # add the {user:click seq}
        test_b.append((user,item))
    with open('split_data/ali_user_seq_'+dat[5:]+'.pkl','wb') as f:
        pkl.dump(test_b,f)