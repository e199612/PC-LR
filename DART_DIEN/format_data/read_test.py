import pickle as pkl
import glob

filename = glob.glob('*.pkl')
data = pkl.load(open(filename,'rb'))
