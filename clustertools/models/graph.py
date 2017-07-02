from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta
from scipy.linalg import eig

from clustertools.models.distance import KMeans

class SpectralClustering(object):

    def __init__(self, data, n, similarity_measure='gaussian', laplacian='normalized', metric='euclidean',
                 kmeans_params = None, verbose=True, **kwargs):
        '''
        Graph-based Spectral Clustering, normalized cuts algorithm by default (normalized graph Laplacian),
        runs KMeans on the eigenspace data. It is planned to implement arbitrary clustering algorithms
        to work on the lower dimensional eigenspace.
        See https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
        
        Args:
            data: (n,d)-shaped two-dimensional ndarray or graph adjacency matrix
            n: number of clusters to be determined
            similarity measure: specification of similarity measure on data array for graph generation
                'eps_dist' for discrete minimal distance measure: optional argument 'eps' must be passed
                'gaussian' for gaussian similarity measure: optional argument 'bandwidth' must be passed
                'kNN' for k-nearest neighbor similarity measure: optional arguments 'k' and 'kNN_mode' must be passed
                    k: number of nearest neighbors to be considered
                    kNN_mode: 'mutual' or 'unilateral', depending
                None: the given data array will be considered as an adjacency array, no further graph computations will happen
            laplacian: type of the graph Laplacian to be eigendecomposed
                'normalized' for classical normalized cuts algorithm
                'standard' for unnormalized graph Laplacian. Faster than normalized cuts algorithm, but usually yields poor results.
                    This is best used in combination with kNN adjacency arrays.
            metric: specification of used metric used in the similarity measures, see scipy.spatial.distance docs
                WARNING: classic spectral clustering is based on the euclidean distance, it is recommended to use this metric.
            kmeans_params: dict containing optional KMeans settings for clustering on low dimensional eigenspace,
                see clustertools.models.distance.KMeans doc. If None, default settings will be applied.
                Note that "data" and "k" parameter of the KMeans instance will always be determined arguments passed to SpectralClustering.
        '''

        if type(data) is list:
            raise NotImplementedError('SpectralClustering is not list compatible yet')
        
        eps = kwargs.get('eps')
        bandwidth = kwargs.get('bandwidth')
        k = kwargs.get('k')
        kNN_mode = kwargs.get('kNN_mode')
        
        self._n = n
        self._data = data
        self._k = k
        self._laplacian = laplacian
        self._similarity_measure = similarity_measure
        self._cluster_labels = None
        self._n_clusters = None
        self._verbose = verbose
        self._metric = metric
        self._eps = eps
        self._bandwidth = bandwidth
        self._kNN_mode = kNN_mode 
        self._kmeans_params = kmeans_params
        
        if self._metric != 'euclidean':
            print('Warning: spectral clustering not initialized with euclidean metric. This results in a purely experimental algorithm!')
            
        
        #check parameters corresponding two chosen similarity measure
        if self._similarity_measure == 'eps_dist' and self._eps is None:
            raise TypeError('Argument "eps" for given similarity metric not found')
            
        if self._similarity_measure == 'gaussian' and self._bandwidth is None:
            raise TypeError('Argument "bandwidth" for given similarity metric not found')

        if self._similarity_measure == 'kNN' and (self._k is None and self._kNN_mode is None):
            raise TypeError('Arguments "k" and "kNN_mode" for given similarity metric not found')
        

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

        if self._verbose:
            print('\n')
            print('KMeans initialization on eigenvectors...')
           
        if self._kmeans_params is None:
            #default KMeans parameters
            cluster_obj = KMeans(data = eigvecs,k=self._n,method='kmeans++',max_iter=300,atol=10**-12,rtol=10**-12,verbose=self._verbose)
        else:
            #pass dict arguments 
            self._kmeans_params["data"] = eigvecs
            self._kmeans_params["k"] = self._n  
            cluster_obj = KMeans(**self._kmeans_params)
        cluster_obj.fit()
        labels = cluster_obj._cluster_labels
        self.cluster_labels = labels

        #main algorithm verbosity
        if self._verbose:
            print ('KMeans terminated. \n')
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
         utput:
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
