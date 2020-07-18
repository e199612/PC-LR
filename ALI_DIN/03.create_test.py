import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import choices
from collections import Counter,OrderedDict

from os import listdir
from os.path import isfile, isdir, join

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

mypath = './split_data'
files = listdir(mypath)
csv_list = []
for f in files:
    fullpath = join(mypath, f)
    if isfile(fullpath):
        if f[-4:]=='.pkl':
            csv_list.append(f)
csv_list.sort()

for file_name in tqdm(csv_list):
    df = pkl.load(open('./split_data/' + file_name,'rb'))
    test = []
    neg = negative_sampling(df,len(df))
    count = 0
    for line in df:
        if len(line[1]) < 2:
            continue
        test.append((line[0],line[1][:-1],(line[1][-1],neg[count])))
        count += 1
    with open('test_data/test_'+file_name[13:-4]+'.pkl','wb') as f:
        pkl.dump(test,f)