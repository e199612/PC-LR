from tqdm import tqdm
import pandas as pd
import pickle as pkl

data = pd.read_csv('../alibaba_din/UB_clear_pv.csv').drop(['Unnamed: 0'], axis=1)

data = data[data['date'] == '2017-11-25']
data.drop_duplicates(subset=['uid', 'iid'], inplace=True)
data.drop(['cid', 'type', 'timestamp', 'date'], inplace=True, axis=1)

data.to_csv('reviews-info', sep='\t', index=False, header=False)