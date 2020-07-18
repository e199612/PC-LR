import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import choices
import random
from collections import Counter,OrderedDict

def negative_sampling(data, num_of_samples):
    all_last_click = []
    for i in data:
        all_last_click.extend(i[1])
    ctr = Counter(all_last_click)
    dict_ctr_freq = {}
    for i in ctr:
        dict_ctr_freq[i] = ctr[i] / len(all_last_click)
    numerator = { pid: freq ** 0.75 for pid, freq in dict_ctr_freq.items()}
    denominator = sum(numerator.values())
    prob = { pid: freq_f/denominator for pid, freq_f in numerator.items()}
    return choices(list(prob.keys()), weights=list(prob.values()),k=num_of_samples)

data = pkl.load(open('split_data/ali_user_seq_11-25.pkl','rb'))

# need ite-info in dien experiments
i_c = pd.read_csv('../ALI_DIEN/item-info',header=None)

item_cat = {}
item_cat = {i.strip().split('\t')[0]:i.strip().split('\t')[1] for i in tqdm(i_c[0])}

max_i = 0
max_c = 0
for k,v in item_cat.items():
    if int(k) > max_i:
        max_i = int(k)
    if int(v) > max_c:
        max_c = int(v)
    if int(v) == 0:
        print(123)
    if int(k) == 0:
        print(456)

item_cat['UNK'] = 0

with open('ali_i_c.pkl','wb') as f:
    pkl.dump(item_cat,f)

# negative sampling
neg = negative_sampling(data,len(data))

# train adn test
train = []
test = []
count = 0
short_len = 0
for i in tqdm(data):
    if len(i[1]) < 2:
        short_len += 1
        continue
    num = random.randint(1,10)
    if num == 2:
        test.append((i[0],i[1][:-1],(i[1][-1],neg[count])))
        count += 1
    else:
        train.append((i[0],i[1][:-1],i[1][-1],1))
        train.append((i[0],i[1][:-1],neg[count],0))
        count += 1

# create count and category list
cate_list = []
for i in tqdm(range(max_i+1)):
    if str(i) in item_cat:
        cate_list.append(int(item_cat[str(i)]))
    else:
        cate_list.append(0)

item_count = max_i+1
cate_count = max_c+1

max_u = 0
for i in data:
    if i[0] > max_u:
        max_u = i[0]
user_count = max_u+1

with open('ali_new4days_train.pkl','wb') as d:
    pkl.dump([train,test,cate_list,user_count,item_count,cate_count],d)