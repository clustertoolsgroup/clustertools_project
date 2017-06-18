from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta
from scipy.linalg import eig

from clustertools.models.distance import KMeans

class SpectralClustering(object):

    def __init__(self,data,k,similarity_metric='eps_dist',laplacian='standard',metric='euclidean',
                 low_dim_clustering = None,verbose=True,**kwargs):
        '''
        Graph-based Spectral Clustering
        Args:
            data: (n,d)-shaped two-dimensional ndarray
            similarity metric: specification of similarity measure on data
                'eps_dist' for discrete minimal distance measure: optional argument 'eps' must be passed
                'gaussian' for gaussian similarity measure: optional argument 'bandwidth' must be passed
            metric: specification of used metric, see scipy.spatial.distance docs
                WARNING: classic spectral clustering is based on the euclidean distance, it is recommended not to 
                change the used metric
        '''

        if type(data) is list:
            raise NotImplementedError('SpectralClustering is not list compatible yet')
        
        eps = kwargs.get('eps')
        bandwidth = kwargs.get('bandwidth')
        
        self._data = data
        self._k = k
        self._laplacian = laplacian
        self._similarity_metric = similarity_metric
        self._cluster_labels = None
        self._n_clusters = None
        self._verbose = verbose
        self._metric = metric
        self._eps = eps
        self._bandwidth = bandwidth
        self._low_dim_clustering = low_dim_clustering
        
        if self._metric != 'euclidean':
            print('Warning: spectral clustering initialized with different metric than euclidean metric.')
            
        if self._similarity_metric == 'eps_dist' and self._eps is None:
            raise TypeError('Argument "eps" for given similarity metric not found')
            
        if self._similarity_metric == 'gaussian' and self._bandwidth is None:
            raise TypeError('Argument "bandwidth" for given similarity metric not found')
        

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
        runs spectral clustering on data.
        '''

        if self._verbose:
            start_time = timer()

        # initialize variables
        [n_samples,dim] = self._data.shape
        #visited = np.zeros(n_samples,dtype=bool)
        cluster_labels = [None]*n_samples
        #cluster_index = 0
        #noise_counter = 0

        #----------------------------------------------------
        
        #compute adjacency array
        distances = distance.cdist(self._data,self._data,metric=self._metric)
        if self._similarity_metric == 'eps_dist':
            print('Constructing discrete similarity matrix')
            W = (distances < self._eps).astype(int)
        if self._similarity_metric == 'gaussian':
            print('Constructing gaussian similarity matrix')
            W = np.exp(-1*np.power(distances,2)/(2*self._bandwidth*2))
            
        
        D = np.diag(np.sum(W,axis=0))
        
        if self._laplacian == 'standard':
            print('Computing standard Laplacian eigenproblem')
            L = D-W
            eigvals,eigvecs = eig(L)
        if self._laplacian == 'normalized':
            print('Computing generalized Laplacian eigenproblem')
            L = D-W
            eigvals,eigvecs = eig(L,D)
            
        #if self._low_dim_clustering is None:
        # self.low_dim_clustering = KMeans(verbose = False)
            
        eigvecs = dim*(eigvecs)[:,0:self._k]
        
        
        cluster_obj = KMeans(data = eigvecs,k=self._k,method='kmeans++',max_iter=300,atol=10**-12,rtol=10**-12)
        cluster_obj.fit()
        labels = cluster_obj._cluster_labels
        self.cluster_labels = labels

        if self._verbose:
            #print('Detected %i clusters'%cluster_index)
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after ' + str(elapsed_time))
            #noise_rate = noise_counter/n_samples
            #print('Rate of noise in dataset: %f'%noise_rate)
        #self._n_clusters = cluster_index

