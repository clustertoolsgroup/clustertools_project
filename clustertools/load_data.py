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

def load_aggregation_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/Aggregation.txt')
    return data

def load_birch1_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/birch1.txt')
    return data

def load_birch3_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/birch3.txt')
    return data

def load_compound_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/Compound.txt')
    return data

def load_flame_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/flame.txt')
    return data

def load_pathbased_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/pathbased.txt')
    return data

def load_sets_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/\s4.txt')
    return data

def load_spiral_data():
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    data = np.loadtxt('./clustertools/data/spiral.txt')
    return data
