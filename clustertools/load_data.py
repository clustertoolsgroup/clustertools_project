'''
supplies loading functions from example datasets stored in 
./clustertools/data
'''

import numpy as np

def load_fuzzy_data():
	'''
	loads dataset from
	
	https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/clusterable_data.npy
	'''
	return np.load('./clustertools/data/fuzzy_data.npy')
