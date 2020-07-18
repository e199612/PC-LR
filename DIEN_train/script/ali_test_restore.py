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

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")
test_file = os.path.join(get_data_prefix(), "test_11-26_1.pkl")
uid_voc = os.path.join(get_data_prefix(), "uid_voc_4days.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc_4days.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc_4days.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')

item_c = open(item_info,'r')

def test(train_file=train_file,
         test_file=test_file,
         uid_voc=uid_voc,
         mid_voc=mid_voc,
         cat_voc=cat_voc,
         item_info=item_info,
         reviews_info=reviews_info,
         batch_size=99,
         maxlen=100):
    
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

    saver = xdl.Saver()
    checkpoint_version = "ckpt-...............20000"
    saver.restore(version = checkpoint_version)
    
    last_hist = []
    target_list = []
    seq = []
    test_set = pkl.load(open(test_file,'rb'))
    knn_table = pkl.load(open('../data/ali_knn_table/knn'+str(test_file[-5])+'_no_pro2.pkl','rb'))
    print('length before deal with : ',len(test_set))
    test_knn = open('../data/test_knn','w')
    count22 = 0
    for i in test_set:
        # knn
        ss = i.strip().split('\t')
        last = ss[4].split('/')[-1]

        # append last, target, and seq
        last_hist.append(last)
        target_list.append(ss[2])
        seq.append((ss[1],ss[4])) # uid and hist
        knn = knn_table[last]
        
        for k in knn:
            count22 += 1
            if k in i_c:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+i_c[k]+'\t'+ss[4]+'\t'+ss[5]
            else:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+'UNK'+'\t'+ss[4]+'\t'+ss[5]
            print >> test_knn,tmp

    test_knn.close()
    
    print('after last_hist :',len(last_hist))
    print('all test_knn length :',count22)

    # sample_io
    test_knn_f = os.path.join(get_data_prefix(), 'test_knn')
    sample_io = SampleIO(train_file, test_knn_f, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)

    print('all length:',len(last_hist))
    
    test_ops = model.build_final_net(EMBEDDING_DIM, sample_io, is_train=False)
    print('='*10+'start test'+'='*10)
    eval_sess = xdl.TrainSession()
    pro_all, test_auc, loss_sum, accuracy_sum, aux_loss_sum = eval_model(eval_sess, test_ops)
    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, loss_sum, accuracy_sum, aux_loss_sum))

    print ('after pro length :',len(pro_all))
    print('='*50)

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

    # print(rank_all_knn)
    # save the result of re-rank
    user = [i[0] for i in seq]
    hist = [i[1] for i in seq]
    assert len(last_hist) == len(user)
    results = list(zip(user, hist, last_hist, target_list, rank))
    # results = pd.DataFrame(results, columns = ['last','target','rank'])
    # esults.to_csv('ali_dien_rank.csv',index=False)
    with open('ali_dien_rank_4days' + test_file[-11:],'wb') as d:
        pkl.dump(results,d)

    # with open('ali_dien_99_' + test_file[-11:],'wb') as d:
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
        filename = glob.glob('../data/ali_dien_test_data/*.pkl')
        filename.sort()
        print(filename)
        for n in filename[16:]:
            print(n)
            test(test_file=n)
    else:
        print('job type must be train or test, do nothing...')