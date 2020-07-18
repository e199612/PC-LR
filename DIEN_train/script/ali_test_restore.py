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

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")
test_file = os.path.join(get_data_prefix(), "test_11-26_1.pkl")
uid_voc = os.path.join(get_data_prefix(), "uid_voc.pkl")
mid_voc = os.path.join(get_data_prefix(), "mid_voc.pkl")
cat_voc = os.path.join(get_data_prefix(), "cat_voc.pkl")
item_info = os.path.join(get_data_prefix(), 'item-info')
reviews_info = os.path.join(get_data_prefix(), 'reviews-info')

# knn_key = pkl.load(open('../data/ali_knn_table/ali_knn_key2.pkl','rb'))
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
    checkpoint_version = "ckpt-................8000"
    saver.restore(version = checkpoint_version)
    
    last_hist = []
    target_list = []
    seq = []
    test_set = pkl.load(open(test_file,'rb'))
    knn_table = pkl.load(open('../data/ali_knn_table/knn'+str(test_file[-5])+'_no_pro2.pkl','rb'))
    print('before length :',len(test_set))
    test_knn = open('../data/test_knn','w')
    for i in test_set[:(len(test_set)//2)]:
        # knn
        ss = i.strip().split('\t')
        last = ss[4].split('/')[-1]

        # append last, target, and seq
        last_hist.append(last)
        target_list.append(ss[2])
        seq.append((ss[1],ss[4])) # uid and hist
        knn = knn_table[last]
        
        for k in knn:
            if k in i_c:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+i_c[k]+'\t'+ss[4]+'\t'+ss[5]
            else:
                tmp = '1\t'+ss[1]+'\t'+k+'\t'+'UNK'+'\t'+ss[4]+'\t'+ss[5]
            print >> test_knn,tmp
    test_knn.close()
    print('finish test data!')

    # sample_io
    test_knn_f = os.path.join(get_data_prefix(), 'test_knn')
    sample_io = SampleIO(train_file, test_knn_f, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)

    print('all length:',len(last_hist))
    
    # test
    # datas = sample_io.next_test()
    # test_ops = tf_test_model(*model.xdl_embedding(datas, EMBEDDING_DIM, *sample_io.get_n()))
    # print('='*10,'start test','='*10)
    test_ops = model.build_final_net(EMBEDDING_DIM, sample_io, is_train=False)
    print('='*10+'start test'+'='*10)
    eval_sess = xdl.TrainSession()
    pro_all, test_auc, loss_sum, accuracy_sum, aux_loss_sum = eval_model(eval_sess, test_ops)
    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (test_auc, loss_sum, accuracy_sum, aux_loss_sum))

    print('pro length :',len(pro_all))
    print('='*50)

    # sort the knn with prob
    rank = []
    for i in range(len(last_hist)):
        # read knn
        knn = knn_table[last_hist[i]]
        pro = pro_all[i]

        c = list(zip(knn,pro))
        c = sorted(c, key = lambda t: t[1], reverse=True)
        count = False
        rerank = 1
        for k in c:
            if target_list[i] == k[0]:
                rank.append(rerank)
                count = True
                break
            rerank += 1
        if count == False:
            rank.append(100)

    # save the result of re-rank
    user = [i[0] for i in seq]
    hist = [i[1] for i in seq]
    results = list(zip(user, hist, last_hist, target_list, rank))
    #results = pd.DataFrame(results, columns = ['last','target','rank'])
    #results.to_csv('ali_dien_rank.csv',index=False)
    with open('ali_dien_rank_' + test_file[-11:-6],'wb') as d:
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