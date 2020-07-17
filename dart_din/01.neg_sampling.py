import pickle as pkl
from tqdm import tqdm
from random import choices
import pandas as pd
from collections import Counter,OrderedDict

from os import listdir
from os.path import isfile, isdir, join

def negative_sampling(data, num_of_samples):
    all_last_click = []
    for i in data:
        all_last_click.append(i[2])
        all_last_click.extend(i[1])
    ctr = Counter(all_last_click)
    dict_ctr_freq = {}
    for i in ctr:
        dict_ctr_freq[i] = ctr[i] / len(all_last_click)
    numerator = { pid: freq ** 0.75 for pid, freq in dict_ctr_freq.items()}
    denominator = sum(numerator.values())
    prob = { pid: freq_f/denominator for pid, freq_f in numerator.items()}
    return choices(list(prob.keys()), weights=list(prob.values()),k=num_of_samples)

def encode(seq,vocabb):
    tmp = []
    for w in seq:
        if w in vocabb:
            tmp.append(vocabb[w])
        else:
            print('what!!!!!')
            break
    return tmp

def item_cat_find(click):
    li = []
    for i in click:
        if i in item_cat:
            li.append(item_cat[i])
        else:
            li.append(item_cat[-1])
    return li

def train_neg(train_data):
    tmppp = negative_sampling(train_data,len(train_data))
    assert len(tmppp) == len(train_data)
    count2 = 0
    train_2 = []
    for i in train_data:
        train_2.append((i[0],i[1],tmppp[count2],0))
        count2 += 1
    train_data = train_data + train_2
    
    return train_data

def test_neg(test_data):
    # negative sampling for testing data
    # put last element of click stream as negative sampling
    tmp2 = negative_sampling(test_data,len(test_data))
    assert len(tmp2) == len(test_data)
    count3 = 0
    test_set = []
    for i in test_data:
        test_set.append((i[0],i[1],(i[2],tmp2[count3])))
        count3 += 1
    
    return test_set

mypath = './pid_corpus'
files = listdir(mypath)
csv_list = []
for f in files:
      fullpath = join(mypath, f)
      # 判斷 fullpath 是檔案還是目錄
      if isfile(fullpath):
        if f[-4:]=='.pkl':
            csv_list.append(f)
csv_list.sort()

train_pkl = csv_list[:14]
test_pkl = csv_list[68:]

# load vocab
item_voc = pd.read_csv(open('vocab/change_item_vocab.txt'),header=None)
cate_voc = pd.read_csv(open('vocab/change_cate_vocab.txt'),header=None)
user_voc = pd.read_csv(open('vocab/change_user_vocab.txt'),header=None)
# prepare the tokenization vocab
i_vocab = list(item_voc[0])
ii_vocab = OrderedDict(zip(i_vocab, range(1,len(i_vocab)+1)))
c_vocab = list(cate_voc[0])
cc_vocab = OrderedDict(zip(c_vocab, range(1,len(c_vocab)+1)))
u_vocab = list(user_voc[0])
uu_vocab = OrderedDict(zip(u_vocab, range(1,len(u_vocab)+1)))

item_cat = pkl.load(open('vocab/change_item_cat.pkl','rb'))

# build the training data format
for file_name in train_pkl:
    print(file_name)
    df = pkl.load(open('./pid_corpus/' + file_name,'rb'))
    train_set = []
    train_data_cat = []
    for u,s in tqdm(df.items()):
        # item cat
        category = item_cat_find(s)

        # encode item
        uid = encode([u],uu_vocab)
        stream = encode(s,ii_vocab)
        cat_en = encode(category,cc_vocab)
        assert len(stream) == len(cat_en)
        train_data_cat.append(cat_en)

        # split into hist and target
        hist = stream[:-1]
        target = stream[-1]
        if len(hist) > 100:
            count += 1
        train_set.append((uid,hist,target,1))
            
    if file_name in train_pkl:
        train_data = train_neg(train_set)
        with open('./preprocess_data/' + 'data_' + file_name[11:-4] + '.pkl', 'wb') as file:
            pkl.dump(train_data, file)
    else:
        train_data = test_neg(train_set)
        with open('./preprocess_data/' + 'data_test_' + file_name[11:-4] + '.pkl', 'wb') as file:
            pkl.dump(train_data, file)