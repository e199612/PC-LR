import pickle as pkl
from tqdm import tqdm
from random import choices
import pandas as pd
from collections import Counter,OrderedDict

from os import listdir
from os.path import isfile, isdir, join

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

fo = open("item-info", "w")
fr = open("reviews-info", "w")
i_c = {}

for file_name in tqdm(csv_list):
    df = pkl.load(open('format_data/' + file_name,'rb'))
    for line in df:
        ls = line.strip().split('\t')
        # item and cate
        iid = ls[4].split('/')
        cid = ls[5].split('/')
        for i in range(len(iid)):
            i_c[iid[i]] = cid[i]
        # user's click hist
        print(ls[1]+'\t'+ls[2],file=fr,end='\n')
        for i in iid:
            print(ls[1]+'\t'+i,file=fr,end='\n')

for k,v in tqdm(i_c.items()):
    print(k+'\t'+v,file=fo,end='\n')
