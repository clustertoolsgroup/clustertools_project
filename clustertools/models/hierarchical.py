from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta

class HierarchicalClustering(object):

    def __init__(self, data, link='average', num_stop_crit=1, metric='euclidean',  verbose=True, **kwargs):
        
        '''
        Hierachical clustering, see https://en.wikipedia.org/wiki/Hierarchical_clustering
        
        TODO:

        Args:
            data: (n,d)-shaped two-dimensional ndarray or distance matrix
            link: specification of the linkage critetion
                'minimum' or 'single' for single linkage (clusters with smallest pointwise minimum distance will be linked)
                'maximum' or 'complete' for complete linkage (clusters with smallest pointwise maximum distance will be linked)
                'average' for average linkage (clusters with smallest pointwise average distance will be linked)
            num_stop_crit=1: number of stopping criteria to be fulfilled to terminate the algorithm: optional arguments 'k' or 'stop_dist' must be passed
                k: number of clusters to produce (will link clusters, until k clusters are remaining)
                stop_dist: stop, when linkage criterion distance between clusters exceeds the given stopping distance stop_dist
            metric:
                either: specification of used metric used in the similarity measures, see scipy.spatial.distance docs
                WARNING: classic clustering is based on the euclidean distance, it is recommended to use this metric.
                or: None: the given data array will be considered as the distance matrix already (and not as positions)
        '''
        
        k = kwargs.get('k')
        stop_dist = kwargs.get('stop_dist')
        
        self._data = data
        self._link = link
        self._num_stop_crit
        self._k = k
        self._stop_dist = stop_dist
        self._cluster_labels = None
        self._verbose = verbose
        self._metric = metric
        
        if all((self._metric != 'euclidean', self._metric is not None)):
            print('Warning: hierachical clustering not initialized with euclidean metric or None. This results in a purely experimental algorithm!')
            
        if self._num_stop_crit > 2
            raise NotImplementedError('currently, just 2 stopping criteria for the algorithm are possible (stop_dist and k)')
        
        # check, if at least one stopping parameter and also enough stopping parameters are given according to num_stop_crit
        if self._num_stop_crit <= 0:
            raise InvalidValue('number of given stopping criteria must be greater than zero')
        num_given_crit = 0
        if self._k is not None:
            num_given_crit += 1
        if self._stop_dist is not None:
            num_given_crit += 1
        if self._num_stop_crit > num_given_crit:
            raise InvalidValue('more stopping criteria were given than specified stopping parameters')
        
        ##################        
        ##################
        ### CHECKPOINT ###
        ##################
        ##################

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
        Runs spectral clustering on data.
        '''

        if self._verbose:
           start_time = timer()

        [n_samples,dim] = self._data.shape
        cluster_labels = [None]*n_samples
        
        #----------------------------------------------------
        
        #compute adjacency array
        if self._similarity_measure is not None:
            distances = distance.cdist(self._data,self._data,metric=self._metric)
            if self._similarity_measure == 'eps_dist':
                print('Constructing discrete similarity matrix')
                W = self.eps_dist_adjacency(distances,self._eps)
            if self._similarity_measure == 'gaussian':
                print('Constructing gaussian similarity matrix')
                W = np.exp(-1*np.power(distances,2)/(2*self._bandwidth**2))
            if self._similarity_measure == 'kNN':
                print('Constructing kNN adjacency matrix')
                W = self.kNN_adjacency(self._k,distances,mode = self._kNN_mode)
        else:
            W = self._data

        #graph degree matrix
        D = np.diag(np.sum(W,axis=0))
       
        #construct laplacian matrix
        if self._laplacian == 'standard':
            print('Computing standard Laplacian eigenproblem')
            L = D-W
            eigvals,eigvecs = eig(L)
        if self._laplacian == 'normalized':
            print('Computing generalized Laplacian eigenproblem')
            L = D-W
            eigvals,eigvecs = eig(L,D)
            
        #rescaling of eigenvectors for computational purposes
        eigvecs = dim*(eigvecs)[:,0:self._n]
        
        
        #------------------
        #lowdimensional KMeans run on eigenvectors

        #if self._verbose:
        #    print('\n')
        #    print('KMeans initialization on eigenvectors...')

        #TODO optional KMeans arguments for clustering on reduced eigenspace
        cluster_obj = KMeans(data = eigvecs,k=self._n,method='kmeans++',max_iter=300,atol=10**-12,rtol=10**-12,verbose=self._verbose)
        cluster_obj.fit()
        labels = cluster_obj._cluster_labels
        self.cluster_labels = labels

        #main algorithm verbosity
        if self._verbose:
            #print ('KMeans terminated. \n')
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after ' + str(elapsed_time))

    #---------------------------------
    #adjacency matrix generation methods
    #---------------------------------

    def kNN_adjacency(self,k,distances,mode='mutual'):
        '''
        Constructs unweighted kNN adjacency matrix based on distance matrix generated from data.
    
        input:
            k: int, determines the depth of neighbor-relation in the data
            distances: (n,n)-shaped ndarray, see scipy.spatial.distance.cdist
            mode:
                'mutual': two points are adjacent, if both individually are among the k nearest neighbors of the other
                'unilateral': two points are adjacent, if at least one is among the k nearest neighbors of the other
        output:
            W: (0,1) ndarray adjacency matrix
        '''
        sorted_indices = np.argsort(distances,axis=1)[:,1:k+1]
        dim = distances.shape[0]
        neighbors = np.zeros((dim,dim))
    
        if mode == 'unilateral':
            raise NotImplementedError('Unilateral kNN graph generation not yet implemented')
    
        if mode == 'mutual':
            for i,vec in enumerate(sorted_indices):
                neighbors[i,vec] = 1
    
                W = np.multiply(neighbors,neighbors.T)
        return W

    def eps_dist_adjacency(self,distances,eps):
        '''
        Constructs unweighted epsilon-distance adjacency matrix based on distance matrix generated from data.

        input:
            distances: (n,n)-shaped ndarray, see scipy.spatial.distance.cdist
            eps: float, size of neighborhood-environment around each data point
        output:
            W: (0,1) ndarray adjacency matrix
        '''
        W = (distances < self._eps).astype(int)
        return W
