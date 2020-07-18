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

EMBEDDING_DIM = 8
HIDDEN_SIZE = 8 * 2
ATTENTION_SIZE = 8 * 2
best_auc = 0.0

def get_data_prefix():
    return xdl.get_config('data_dir')

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser_predict")
test_file = os.path.join(get_data_prefix(), "test_20190315.pkl")
uid_voc = os.path.join(get_data_prefix(), "uid_voc.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')

knn_table = pkl.load(open('../data/change_knn_table/dict_knn_table2.pkl','rb'))
item_c = open('../data/item-info','r')

def test(train_file=train_file,
         test_file=test_file,
         uid_voc=uid_voc,
         mid_voc=mid_voc,
         cat_voc=cat_voc,
         item_info=item_info,
         reviews_info=reviews_info,
         batch_size=128,
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
    
    last_hist = []
    target_list = []
    rank = []
    test_file = pkl.load(open(test_file,'rb'))
    count222 = 0
    for i in test_file:
        print('count :',count222)
        # knn
        ss = i.strip().split('\t')
        last = ss[4].split('/')[-1]

        # find knn
        # print('find knn !!!!')
        if last in knn_table:
            knn = knn_table[last]
        else:
            continue

        # check last has knn
        last_hist.append(last)
        target_list.append(ss[2])

        test_knn = open('../data/test_knn_new','w')
        for k in knn:
            if k in i_c:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+i_c[k]+'\t'+ss[4]+'\t'+ss[5]
            else:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+'UNK'+'\t'+ss[4]+'\t'+ss[5]
            print >> test_knn,tmp
        test_knn.close()

        # sample_io
        # print('sample io !!!!!')
        test_file = os.path.join(get_data_prefix(), "test_knn_new")
        sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc,cat_voc, item_info, reviews_info,batch_size, maxlen, EMBEDDING_DIM)
        # sample_io = SampleIO(train_file, 'test_knn', uid_voc, mid_voc, cat_voc, batch_size, maxlen, EMBEDDING_DIM)

        # test
        # datas = sample_io.next_test()
        # test_ops = tf_test_model(*model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
        # print('='*10,'start test','='*10)
        test_ops = model.build_final_net(EMBEDDING_DIM, sample_io, is_train=False)
        print('='*10+'start predict'+'='*10)
        saver = xdl.Saver()
        checkpoint_version = "ckpt-................4000"
        saver.restore(version = checkpoint_version)
        eval_sess = xdl.TrainSession()
        pro_all, test_auc, loss_sum, accuracy_sum, aux_loss_sum = eval_model(eval_sess, test_ops)
        assert len(pro_all) == 99
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, loss_sum, accuracy_sum, aux_loss_sum))
        count222 += 1
        
        # sort the knn with prob
        c = list(zip(knn,pro_all))
        c.sort(key = lambda t: t[1], reverse=True)
        count = False
        rerank = 1
        for i in c:
            if ss[2] == i[0]:
                rank.append(rerank)
                count = True
                break
            rerank += 1
        if count == False:
            rank.append(100)

        # if count222 == 40:
        #     break

    # save the result of re-rank
    results = list(zip(last_hist,target_list,rank))
    # results = pd.DataFrame(results, columns = ['last','target','rank'])
    # results.to_csv('change_dien_rank.csv',index=False)
    with open('change_dien_rank.pkl','wb') as d:
        pkl.dump(results,d)
    



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
        test()
    else:
        print('job type must be train or test, do nothing...')