import pandas as pd
import numpy as np
import datetime
import pickle as pkl
from tqdm import tqdm
from collections import Counter,OrderedDict
from os import listdir
from os.path import isfile, isdir, join

def encode(seq,vocabb):
    tmp = []
    for w in seq:
        if w in vocabb:
            tmp.append(vocabb[w])
        else:
            print('what!!!!!')
            break
    return tmp

mypath = './preprocess_data'
files = listdir(mypath)
csv_list = []
for f in files:
    fullpath = join(mypath, f)
    if isfile(fullpath):
        if f[-4:]=='.pkl':
            csv_list.append(f)
csv_list.sort()

train_data = []
for file_name in tqdm(csv_list[:55]):
    df = pkl.load(open('./preprocess_data/' + file_name,'rb'))
    for seq in df:
        train_data.append((seq[0][0],seq[1],seq[2],seq[3]))

test_data = []
for file_name in tqdm(csv_list[109:115]):
    df = pkl.load(open('./preprocess_data/' + file_name,'rb'))
    for seq in df:
        test_data.append((seq[0][0],seq[1],seq[2]))

item_ = pd.read_csv('vocab/change_item_vocab.txt',header=None)
item_count = len(list(item_[0]))
cate_ = pd.read_csv('vocab/change_cate_vocab.txt',header=None)
cate_count = len(list(cate_[0]))
user_ = pd.read_csv('vocab/change_user_vocab.txt',header=None)
user_count = len(list(user_[0]))

item_cat = pkl.load(open('vocab/change_item_cat.pkl','rb'))
cate_list = [item_cat[i] for i in item_cat]
c_vocab = list(cate_[0])
cc_vocab = OrderedDict(zip(c_vocab, range(1,len(c_vocab)+1)))
cate_list_real = encode(cate_list,cc_vocab)

with open('chang_new_split_dataset_2.pkl','wb') as f:
    pkl.dump((train_data,test_data,cate_list_real,user_count,item_count,cate_count),f)
