import pickle as pkl
from tqdm import tqdm
from random import choices
import pandas as pd
from collections import Counter,OrderedDict

from os import listdir
from os.path import isfile, isdir, join

def negative_sampling(data, num_of_samples):
    ctr = Counter(data)
    dict_ctr_freq = {}
    for i in ctr:
        dict_ctr_freq[i] = ctr[i] / len(data)
    numerator = { pid: freq ** 0.75 for pid, freq in dict_ctr_freq.items()}
    denominator = sum(numerator.values())
    prob = { pid: freq_f/denominator for pid, freq_f in numerator.items()}
    return choices(list(prob.keys()), weights=list(prob.values()),k=num_of_samples)

i_c = pkl.load(open('../change_din/vocab/change_item_cat.pkl','rb'))

mypath = 'format_data/'
files = listdir(mypath)
csv_list = []
for f in files:
      fullpath = join(mypath, f)
      # 判斷 fullpath 是檔案還是目錄
      if isfile(fullpath):
        if f[-4:]=='.pkl':
            csv_list.append(f)
csv_list.sort()

train_pkl = csv_list[:61]
test_pkl = csv_list[68:]

data = []
for file_name in tqdm(train_pkl):
    df = pkl.load(open('format_data/' + file_name,'rb'))
    data.extend(df)

# extract all clicked item
all_item = []
for i in data:
    h = i.strip().split('\t')[4].split('/')
    all_item.extend(h)

# sample the negative item
neg_list = negative_sampling(all_item,len(data))
assert len(neg_list) == len(data)

# train dataset sample
neg = 0
ftrain = open("train_data_61", "w")
for file_name in tqdm(train_pkl):
    df = pkl.load(open('format_data/' + file_name,'rb'))
    length = len(df)
    for i in range(length):
        ii = df[i].strip().split('\t')
        tmp = '0\t'+ii[1]+'\t'+neg_list[neg]+'\t'+i_c[neg_list[neg]]+'\t'+ii[4]+'\t'+ii[5]
        neg += 1
        print(tmp,end='\n',file=ftrain)
        print(df[i],end='\n',file=ftrain)

# test dataset sample
ftest = open("local_test_splitByUser_61", "w")
for file_name in tqdm(test_pkl):
    df = pkl.load(open('format_data/' + file_name,'rb'))
    length = len(df)
    neg_list = negative_sampling(all_item,length)
    neg = 0
    for i in range(length):
        ii = df[i].strip().split('\t')
        tmp = '0\t'+ii[1]+'\t'+neg_list[neg]+'\t'+i_c[neg_list[neg]]+'\t'+ii[4]+'\t'+ii[5]
        neg += 1
        print(tmp,end='\n',file=ftest)
        print(df[i],end='\n',file=ftest)