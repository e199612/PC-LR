# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pickle as pkl

f_train = open("train_data_61", "r")
uid_dict = {}
mid_dict = {}
cat_dict = {}

for line in f_train:
    arr = line.strip("\n").split("\t")
    clk = arr[0]
    uid = arr[1]
    mid = arr[2]
    cat = arr[3]
    mid_list = arr[4]
    cat_list = arr[5]
    if uid not in uid_dict:
        uid_dict[uid] = 0
    uid_dict[uid] += 1
    if mid not in mid_dict:
        mid_dict[mid] = 0
    mid_dict[mid] += 1
    if cat not in cat_dict:
        cat_dict[cat] = 0
    cat_dict[cat] += 1
    if len(mid_list) == 0:
        continue
    for m in mid_list.split("/"):
        if m not in mid_dict:
            mid_dict[m] = 0
        mid_dict[m] += 1
    for c in cat_list.split("/"):
        if c not in cat_dict:
            cat_dict[c] = 0
        cat_dict[c] += 1

sorted_uid_dict = sorted(uid_dict.items(), key=lambda x:x[1], reverse=True)
sorted_mid_dict = sorted(mid_dict.items(), key=lambda x:x[1], reverse=True)
sorted_cat_dict = sorted(cat_dict.items(), key=lambda x:x[1], reverse=True)

uid_voc = {}
index = 0
for key, value in sorted_uid_dict:
    uid_voc[key] = index
    index += 1

mid_voc = {}
mid_voc["UNK"] = 0
index = 1
for key, value in sorted_mid_dict:
    if key == 'UNK':
        continue
    mid_voc[key] = index
    index += 1

cat_voc = {}
cat_voc["UNK"] = 0
index = 1
for key, value in sorted_cat_dict:
    if key == 'UNK':
        continue
    cat_voc[key] = index
    index += 1

pkl.dump(uid_voc, open("uid_voc_61.pkl", "wb"), 2)
pkl.dump(mid_voc, open("mid_voc_61.pkl", "wb"), 2)
pkl.dump(cat_voc, open("cat_voc_61.pkl", "wb"), 2)
