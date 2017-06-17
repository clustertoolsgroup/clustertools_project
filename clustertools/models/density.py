from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta

class DBSCAN(object):

    def __init__(self,data,eps,minPts,metric='euclidean',verbose=True):
        '''
        Classic density based spatial clustering with noise classification.
        Args:
            data: (n,d)-shaped two-dimensional ndarray
            eps: epsilon neighborhood parameter
            minPts: minimal number of points in each neighborhood
        '''

        if type(data) is list:
            raise NotImplementedError('DBSCAN is not list compatible yet')

        self._data = data
        self._eps = eps
        self._minPts = minPts
        self._cluster_labels = None
        self._metric = metric
        self._n_clusters = None
        self._verbose = verbose

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
                self.fit()
        return self._cluster_labels
    @cluster_labels.setter
    def cluster_labels(self,value):
        self._cluster_labels = value

    def fit(self):
        '''
        classifies the data with DBSCAN algorithm
        '''

        if self._verbose:
            start_time = timer()

        # initialize variables
        [n_samples,dim] = self._data.shape
        visited = np.zeros(n_samples,dtype=bool)
        cluster_labels = [None]*n_samples
        cluster_index = 0
        noise_counter = 0

        for i,observation in enumerate(self._data):
            if visited[i]:
                pass
            else:
                visited[i] = True
                neighbor_indices = get_region(self._data,observation,self._eps,self._metric)
                if len(neighbor_indices) < self._minPts:
                    # mark as noise
                    cluster_labels[i] = 'noise'
                    noise_counter = noise_counter + 1
                else:
                    # move up to next cluster
                    cluster_index = cluster_index + 1
                    #-------------
                    #expand cluster subalgorithm
                    #-------------
                    cluster_labels,visited=expand_cluster(self._data,i,neighbor_indices,cluster_labels,cluster_index,self._eps,self._minPts,visited,self._metric)


        self.cluster_labels = cluster_labels
        if self._verbose:
            print('Detected %i clusters'%cluster_index)
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after ' + str(elapsed_time))
            noise_rate = noise_counter/n_samples
            print('Rate of noise in dataset: %f'%noise_rate)
        self._n_clusters = cluster_index


#------------
#global functions
#------------

def get_region(data,p,eps,metric):
    '''
    returns subset of data containing all points in the eps-ball around p with respect to given metric and the
    corresponding indices
    '''
    n_samples,dim  = data.shape
    distances = distance.cdist(p.reshape(1,dim),data,metric=metric)
    mask = distances<eps
    mask = mask.reshape((n_samples,))
    indices = np.arange(n_samples)[mask]
    #region = data[mask]

    return indices

def expand_cluster(data,i,neighbor_indices,cluster_labels,active_cluster_index,eps,minPts,visited,metric):
    '''

    '''
    #label active point
    cluster_labels[i] = active_cluster_index
    neighbors = data[neighbor_indices]

    #walk through unvisited points in neighborhood
    while True:
        counter = 0

        #expand neighborhood until it does not grow anymore
        while True:

            k=neighbor_indices[counter]
            if not visited[k]:
                visited[k] = True
                neighbor = data[k]
                neighbor_indices2 = get_region(data,neighbor,eps,metric)
                if len(neighbor_indices2) >= minPts:
                    neighbor_indices = list(set(neighbor_indices)|set(neighbor_indices2))
            if cluster_labels[k] is None:
                cluster_labels[k] = active_cluster_index
            counter = counter + 1
            #break condition inner loop: counter bigger than nbhd size
            if counter>=len(neighbor_indices):
                break

        #break condition outer loop: all points visited
        if np.all(visited[neighbor_indices]):
            break

    return cluster_labels,visited









