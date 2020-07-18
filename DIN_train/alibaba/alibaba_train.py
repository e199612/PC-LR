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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1111)
np.random.seed(1111)
tf.set_random_seed(1111)

train_batch_size = 128
test_batch_size = 128
predict_batch_size = 1
predict_users_num = 100
predict_ads_num = 99

with open('ali_new4days_train.pkl', 'rb') as f:
    train_set, test_set, cate_list, user_count, item_count, cate_count  = pkl.load(f)

max_id = 0
for i in tqdm(train_set):
    for el in i[1]:
        if el > max_id:
            max_id = el
    if i[2] > max_id:
        max_id = el
for k in tqdm(test_set):
    for el in k[1]:
        if el > max_id:
            max_id = el
    for el in k[2]:
        if el > max_id:
            max_id = el
print('max_id : ', max_id)

item_cat = pkl.load(open('alibaba_item_cat2.pkl','rb'))
cate_list = []
for i in tqdm(range(max_id+1)):
    if i in item_cat:
        cate_list.append(item_cat[i])
    else:
        cate_list.append(-1)
cate_list_real = cate_list

max_cat = 0
cat_num = {}
for c in cate_list_real:
    if c in cat_num:
        cat_num[c] += 1
    else:
        cat_num[c] = 1
    
    if c > max_cat:
        max_cat = c
print('max_cat : ',max_cat)

# fill the None category with most popular category
popular = sorted(cat_num.items(),key=lambda cat_num:cat_num[1])[-2][0] # [-1] = -1
cate_list_real = [popular if x == -1 else x for x in cate_list_real]

cate_count = max_cat+1
item_count = max_id+1

# setting model
best_auc = 0.0

def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
    score_p = score[:,0]
    score_n = score[:,1]
    #print "============== p ============="
    #print score_p
    #print "============== n ============="
    #print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr

def _eval(sess, model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval_(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, 'save_path_train_4days/ckpt')
    return test_gauc, Auc

def _test_(sess, model):
    score_arr = []
    predicted_users_num = 0
    for _, uij in DataInputTest(test_set, predict_batch_size): # (u, i, j, hist_i, sl) = (user, pos, neg, hist, max seq length)
        if predicted_users_num >= predict_users_num:
            break
        score_ = model.test(sess, uij)
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_[0],score_

# start training
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
    sys.stdout.flush()
    lr = 1.0
    #start_time = time.time()
    for _ in range(20):
        start_time = time.time()
        random.shuffle(train_set)

        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss

            if model.global_step.eval() % 1000 == 0:
                test_gauc, Auc = _eval(sess, model)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (model.global_epoch_step.eval(), model.global_step.eval(),
                       loss_sum / 1000, test_gauc, Auc))
                sys.stdout.flush()
                loss_sum = 0.0

            if model.global_step.eval() % 336000 == 0:
                lr = 0.1

        print('Epoch %d DONE\tCost time: %.2f' %(model.global_epoch_step.eval(), time.time()-start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()

    print( '%.2f'%(time.time()-start_time))
    
    print('best test_gauc:', best_auc)
    sys.stdout.flush()