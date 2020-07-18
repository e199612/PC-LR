from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import os
import time
import pickle as pkl
import random
import numpy as np
import tensorflow.compat.v1 as tf
import sys
from input_ import DataInput, DataInputTest
from model_ali import Model

import tokenization
import random
from tqdm import tqdm
import pandas as pd

from os import listdir
from os.path import isfile, isdir, join

last = []
target = []
knn_test = []

def _test(sess, model, data):
    auc_sum = 0.0
    predicted_users_num = 0
    knn = [[], [], [], []]
    for _, uij in tqdm(DataInputTest(data, predict_batch_size)):# (u, i, j, hist_i, sl) = (user, pos, neg, hist, max seq length)
        uij, knn_table = filtering(uij) # new uij and knn_table number
        if knn_table > -1:
            knn[knn_table].append(uij)
    print('='*10+'start predict'+'='*10)
    for k in range(4):
        uij_table = knn[k]
        knn_table = pkl.load(open('knn_table/knn'+str(k+1)+'_no_pro.pkl','rb'))
        for uij in tqdm(uij_table):
            last.append(uij[5])
            target.append(uij[1])
            # load the neighbor knn
            last_knn = []
            last_knn.append(knn_table[str(uij[6][0])])
            score_ = model.test(sess, uij, last_knn)
            # store the score
            knn_test.append(score_)
            predicted_users_num += predict_batch_size
    
    return knn

def filtering(uij):
    knn_table = -1
    for i in range(len(knn_key)):
        if str(uij[5][0]) in knn_key[i]:
            knn_table = i
            break
    if knn_table < 0:
        return uij,knn_table
    uij = list(uij)
    tmp = uij[5]
    uij.insert(5,[0]) # because last_knn has only one list
    uij.insert(6,tmp)
    uij = tuple(uij)
    
    return uij, knn_table

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1111)
np.random.seed(1111)
tf.set_random_seed(1111)

train_batch_size = 128
test_batch_size = 128
predict_batch_size = 1
predict_users_num = 100
predict_ads_num = 99

info = pkl.load(open('ali_test_info_4days.pkl','rb'))

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
model = Model(info[0], info[1], info[2], info[3], predict_batch_size, predict_ads_num)
sess.run(tf.global_variables_initializer())
model.restore_(sess, './save_path_alibaba_new/ckpt')

knn_key = pkl.load(open('knn_table/ali_knn_key.pkl','rb'))

mypath = './test_data'
files = listdir(mypath)
csv_list = []
for f in files:
    fullpath = join(mypath, f)
    if isfile(fullpath):
        if f[-4:]=='.pkl':
            csv_list.append(f)
csv_list.sort()


for file_name in csv_list:
    print(file_name)
    df = pkl.load(open('./test_data/' + file_name,'rb'))
    score_test =  _test(sess, model, df)