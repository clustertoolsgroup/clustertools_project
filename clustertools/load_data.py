'''
supplies loading functions from example datasets stored in 
./clustertools/data
'''

import numpy as np

def load_fuzzy_data(scale = 'False'):
    '''
    loads dataset from
    	
    https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/clusterable_data.npy
    '''
    if scale == 'True':
        data = scale_data(np.load('./clustertools/data/fuzzy_data.npy'))
    else:
        data =  np.load('./clustertools/data/fuzzy_data.npy')
    return data

def load_aggregation_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/Aggregation.txt'))
    else:
        data = np.loadtxt('./clustertools/data/Aggregation.txt')
    return data

def load_birch1_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/birch1.txt'))
    else:
        data = np.loadtxt('./clustertools/data/birch1.txt')
    return data

def load_birch3_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/birch3.txt'))
    else:
        data = np.loadtxt('./clustertools/data/birch3.txt')
    return data

def load_compound_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/Compound.txt'))
    else:
        data = np.loadtxt('./clustertools/data/Compound.txt')
    return data

def load_flame_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/flame.txt'))
    else:
        data = np.loadtxt('./clustertools/data/flame.txt')
    return data

def load_pathbased_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/pathbased.txt'))
    else:
        data = np.loadtxt('./clustertools/data/pathbased.txt')
    return data

def load_sets_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/\s4.txt'))
    else:
        data = np.loadtxt('./clustertools/data/\s4.txt')
    return data

def load_spiral_data(scale = 'False'):
    '''
    datasets from: https://cs.joensuu.fi/sipu/datasets/
    '''
    if scale == 'True':
        data = scale_data(np.loadtxt('./clustertools/data/spiral.txt'))
    else: 
        data = np.loadtxt('./clustertools/data/spiral.txt')
    return data

def scale_data(data):
    '''
    scale data to mean 0 and standard deviation 1
    http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
    '''
    [n,d] = np.shape(data)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    scaled = np.array([(data[:,i]-mean[i])/std[i] for i in range(d)]).T
    return scaled

def load_uneven_blobs():
    '''
    uneven blobs dataset generated with sklearn.datasets.make_blobs
    '''
    from sklearn.datasets import make_blobs
    n_samples = 1500
    random_state = 3
    X, y = make_blobs(n_samples=n_samples, cluster_std=[1.5, 2.5, 2.5], random_state=random_state)
    data = np.vstack((X[y == 0][:500], X[y == 1][:200], X[y == 2][:200]))
    return data