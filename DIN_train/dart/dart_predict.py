import os
import time
import pickle as pkl
import random
import numpy as np
import tensorflow.compat.v1 as tf
import sys
from input_ import DataInput, DataInputTest
from model import Model

import tokenization
import random
from tqdm import tqdm
import pandas as pd

from os import listdir
from os.path import isfile, isdir, join

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1111)
np.random.seed(1111)
tf.set_random_seed(1111)
train_batch_size = 32
test_batch_size = 32
predict_batch_size = 1
predict_users_num = 10
predict_ads_num = 99

def _test(sess, model, data_set):
    auc_sum = 0.0
    predicted_users_num = 0
    for _, uij in tqdm(DataInputTest(data_set, predict_batch_size)): # (u, i, j, hist_i, sl, last) = (user, pos, neg, hist, max seq length)
        if len(uij[3]) > 100:
            continue
        seq.append((uij[0],uij[3]))
        last.append(uij[5])
        target.append(uij[1])
        score_, logits = model.test(sess, uij)
        knn_test.append(score_)
        predicted_users_num += predict_batch_size
    return predicted_users_num

info = pkl.load(open('change_test_info_61days.pkl','rb'))

import tensorflow as tf
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

model = Model(info[0], info[1], info[2], info[3], predict_batch_size, predict_ads_num)
sess.run(tf.global_variables_initializer())

model.restore_(sess, './save_path_change_new_split_61days/ckpt')

mypath = './change_test_file'
files = listdir(mypath)
csv_list = []
for f in files:
      fullpath = join(mypath, f)
      # 判斷 fullpath 是檔案還是目錄
      if isfile(fullpath):
        if f[-4:]=='.pkl' and f[:4]=='data':
            csv_list.append(f)
csv_list.sort()

knn = pkl.load(open('item_knn_all_new.pkl','rb'))

for file_name in csv_list:
    print(file_name)
    df = pkl.load(open('./change_test_file/' + file_name,'rb'))
    pred = [i for i in df if len(i[1]) < 100]
    seq = []
    last = []
    target = []
    knn_test = []
    all_rank = []
    num =  _test(sess, model, pred)
    
    # rerank
    target_rank = []
    for i in tqdm(range(len(last))):
        l = last[i][0]
        t = target[i][0]
        k = knn_test[i][0]

        # knn for last item
        knn_ll = knn[l]
        #print(knn_ll)
        
        # concat knn and index
        tryy = pd.DataFrame(k,index=knn_ll, dtype=np.float32)
        tryy = tryy[0].map(lambda x : format(x,'.10f'))
        
        # sort the knn
        tryy.sort_values(ascending=False, inplace=True)
        tryyy = pd.DataFrame(tryy)
        all_rank.append(tryyy)
        tryyy.insert(0,'rank',range(1,100))
        #print(tryyy)
        if t in tryyy.index:
            target_rank.append(tryyy.loc[t]['rank'])
        else:
            target_rank.append(100)
            
        #print(target_rank[0])
    
    # get re rank list
    #print(all_rank[0].index)
    rerank = [list(allr.index) for allr in all_rank]
    
    # create a dataframe
    assert len(seq) == len(target_rank)
    user = [i[0] for i in seq]
    hist = [i[1] for i in seq]
    print(user[0][0])
    cc = list(zip(user,hist,last,target,target_rank))
    pdcc = pd.DataFrame(cc,columns=['uid','hist_iid','last_iid','target_iid','rerank'])
    pdcc = pdcc[pdcc['rerank'] != 100].reset_index(drop=True)
    with open('./change_test_file/' + file_name[5:-4] + '_result_30days.pkl' ,'wb') as f:
        pkl.dump(pdcc,f)