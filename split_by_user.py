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

import random
import pickle as pkl
import pandas as pd

fi = open("train_data_61", "r")
ftrain = open("local_train_splitByUser_61days", "w")
ftest = open("local_test_splitByUser_61days", "w")

while True:
    rand_int = random.randint(1, 10)
    noclk_line = fi.readline().strip()
    clk_line = fi.readline().strip()
    if noclk_line == "" or clk_line == "":
        break
    if rand_int == 2:
        print(noclk_line, file=ftest,end='\n')
        print(clk_line, file=ftest,end='\n')
    else:
        print(noclk_line, file=ftrain,end='\n')
        print(clk_line, file=ftrain,end='\n')
        
'''
fo = pkl.load(open('change_dien_rank.pkl','rb'))
data = pd.DataFrame(fo,columns=['last','target','rank'])
data.to_csv('change_dien_40_results_2.csv',index=False)
for i in fo:
    print(i)
    break
'''