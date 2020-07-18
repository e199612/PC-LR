import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import choices
from collections import Counter,OrderedDict

from os import listdir
from os.path import isfile, isdir, join

def negative_sampling(all_last_click, num_of_samples):
    ctr = Counter(all_last_click)
    dict_ctr_freq = {}
    for i in ctr:
        dict_ctr_freq[i] = ctr[i] / len(all_last_click)
    numerator = { pid: freq ** 0.75 for pid, freq in dict_ctr_freq.items()}
    denominator = sum(numerator.values())
    prob = { pid: freq_f/denominator for pid, freq_f in numerator.items()}
    return choices(list(prob.keys()), weights=list(prob.values()),k=num_of_samples)

data = pkl.load(open('ali_user_seq_4days.pkl','rb'))

# negative sample items
all_item_cat = {}
item = []
cat = []
for i in tqdm(data):
    item.extend(i[1])
    cat.extend(i[2])

for it in range(len(item)):
    all_item_cat[item[it]] = cat[it]

neg_bucket = list(all_item_cat.keys())

# training data
ft = open('ali_train_4days','w')
neg = negative_sampling(neg_bucket,len(data))

count = 0
for el in tqdm(data):
    tmp = '1\t' + str(el[0]) + '\t' + str(el[1][-1]) + '\t' + str(el[2][-1]) + '\t' + '/'.join(str(i) for i in el[1][:-1]) + '\t' + '/'.join(str(i) for i in el[2][:-1])
    tmp2 = '0\t' + str(el[0]) + '\t' + str(neg[count]) + '\t' + str(all_item_cat[neg[count]]) + '\t' + '/'.join(str(i) for i in el[1][:-1]) + '\t' + '/'.join(str(i) for i in el[2][:-1])
    count += 1
    
    print(tmp2, file=ft,end='\n')
    print(tmp, file=ft,end='\n')