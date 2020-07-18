import os
import sys
import time
import math
import random

import tensorflow as tf
import numpy
#import pandas as pd

from model import *
from utils import *
from sample_io import SampleIO
import pickle as pkl

import xdl
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook

import glob

EMBEDDING_DIM = 8
HIDDEN_SIZE = 8 * 2
ATTENTION_SIZE = 8 * 2
best_auc = 0.0

def get_data_prefix():
    return xdl.get_config('data_dir')

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser_predict")
test_file = os.path.join(get_data_prefix(), "test_20190316.pkl")
uid_voc = os.path.join(get_data_prefix(), "uid_voc_61.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc_61.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc_61.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')

knn_table = pkl.load(open('../data/change_knn_table/dict_knn_table2.pkl','rb'))
item_c = open(item_info,'r')

def test(train_file=train_file,
         test_file=test_file,
         uid_voc=uid_voc,
         mid_voc=mid_voc,
         cat_voc=cat_voc,
         item_info=item_info,
         reviews_info=reviews_info,
         batch_size=99,
         maxlen=10000):
    
    if xdl.get_config('model') == 'din':    
        model = Model_DIN(EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    elif xdl.get_config('model') == 'dien':    
        model = Model_DIEN(EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din and dien model')

    # create item cate dict
    i_c = {}
    for i in item_c:
        ii = i.strip().split('\t')
        i_c[ii[0]] = ii[1]
    
    last_hist = []
    target_list = []
    seq = []
    test_set = pkl.load(open(test_file,'rb'))
    test_knn = open('../data/test_knn_new','w')
    print('length before deal with : ',len(test_set))
    count22 = 0
    #count111 = 0
    for i in test_set:
        # knn
        ss = i.strip().split('\t')
        last = ss[4].split('/')[-1]
        
        # find knn
        # check last has knn
        if last in knn_table:
            knn = knn_table[last]
        else:
            continue
        
        # check target in knn
        if ss[2] not in knn:
            continue

        # append last and target
        last_hist.append(last)
        target_list.append(ss[2])
        seq.append((ss[1],ss[4]))

        for k in knn:
            count22 += 1
            if k in i_c:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+i_c[k]+'\t'+ss[4]+'\t'+ss[5]
            else:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+'UNK'+'\t'+ss[4]+'\t'+ss[5]
            print >> test_knn,tmp
        # count111 += 1
        # if count111 == 1000:
        #     break
    test_knn.close()

    print('after last_hist :',len(last_hist))
    print('all test_knn length :',count22)

    # sample_io
    # print('sample io !!!!!')
    test_knn_f = os.path.join(get_data_prefix(), 'test_knn_new')
    sample_io = SampleIO(train_file, test_knn_f, uid_voc, mid_voc,cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)
    # sample_io = SampleIO(train_file, 'test_knn', uid_voc, mid_voc, cat_voc, batch_size, maxlen, EMBEDDING_DIM)

    # test
    test_ops = model.build_final_net(EMBEDDING_DIM, sample_io, is_train=False)

    # start predict
    print('='*10+'start predict'+'='*10)
    saver = xdl.Saver()
    checkpoint_version = "ckpt-...............40000"
    saver.restore(version = checkpoint_version)
    eval_sess = xdl.TrainSession()
    pro_all, test_auc, loss_sum, accuracy_sum, aux_loss_sum = eval_model(eval_sess, test_ops)
    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, loss_sum, accuracy_sum, aux_loss_sum))

    print ('after pro length :',len(pro_all))
    print('='*50)
    assert len(last_hist) == len(target_list), 'Error2!!'
        
    # sort the knn with prob
    rank_all_knn = {}
    rank = []
    for i in range(len(last_hist)):
        knn = knn_table[last_hist[i]]
        pro = pro_all[i]

        c = list(zip(knn,pro))
        c = sorted(c, key = lambda t: t[1], reverse=True)
        rank_all = [sss[0] for sss in c]
        rank_all_knn[seq[i][0]] = rank_all

        if target_list[i] in rank_all:
            rank.append(rank_all.index(target_list[i])+1)
        else:
            rank.append(100)

        # count = False
        # rerank = 1
        # for k in c:
        #     # print(i)
        #     if target_list[i] == k[0]:
        #         rank.append(rerank)
        #         count = True
        #         break
        #     rerank += 1
        # if count == False:
        #     rank.append(100)


    # save the result of re-rank
    user = [i[0] for i in seq]
    hist = [i[1] for i in seq]
    assert len(last_hist) == len(user)
    results = list(zip(user, hist, last_hist, target_list, rank))
    # results = pd.DataFrame(results, columns = ['last','target','rank'])
    # results.to_csv('change_dien_rank.csv',index=False)
    with open('change_dien_rank_61days' + test_file[-12:],'wb') as d:
        pkl.dump(results,d)
    # with open('change_dien_99_' + test_file[-12:],'wb') as d:
    #     pkl.dump(rank_all_knn,d)
    



if __name__ == '__main__':
    SEED = xdl.get_config("seed")
    if SEED is None:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    job_type = xdl.get_config("job_type")
    if job_type == 'train':
        train()
    elif job_type == 'test':
        filename = glob.glob('../data/format_data/test_data_all/*.pkl')
        filename.sort()
        print(filename)
        for n in filename[53:]:
            print(n)
            test(test_file=n)
    else:
        print('job type must be train or test, do nothing...')