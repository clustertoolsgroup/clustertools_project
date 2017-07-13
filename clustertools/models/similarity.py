from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta
from scipy.linalg import eig
from itertools import product

from clustertools.models.distance import KMeans

#----------------------------
#SpectralClustering
#----------------------------

class SpectralClustering(object):

    def __init__(self, data, k, similarity_measure='gaussian', laplacian='normalized', metric='euclidean',
                 kmeans_params = None, verbose=True, **kwargs):
        '''
        Graph-based Spectral Clustering, normalized cuts algorithm by default (normalized graph Laplacian),
        runs KMeans on the eigenspace data. It is planned to implement arbitrary clustering algorithms
        to work on the lower dimensional eigenspace.
        See https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
        
        Args:
            data: (n,d)-shaped two-dimensional ndarray or graph adjacency matrix
            k: number of clusters to be determined
            similarity measure: specification of similarity measure on data array for graph generation
                'eps_dist' for discrete minimal distance measure: optional argument 'eps' must be passed
                'gaussian' for gaussian similarity measure: optional argument 'bandwidth' must be passed
                'kNN' for k-nearest neighbor similarity measure: optional arguments 'k' and 'kNN_mode' must be passed
                    k_neighbor: number of nearest neighbors to be considered
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
        k_neighbor = kwargs.get('k_neighbor')
        kNN_mode = kwargs.get('kNN_mode')
        
        self._k = k
        self._data = data
        self._k_neighbor = k_neighbor
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
                W = self.kNN_adjacency(self._k_neighbor,distances,mode = self._kNN_mode)
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
        eigvecs = dim*(eigvecs)[:,0:self._k]
        
        
        #------------------
        #lowdimensional KMeans run on eigenvectors

        if self._verbose:
            print('\n')
            print('KMeans initialization on eigenvectors...')
           
        if self._kmeans_params is None:
            #default KMeans parameters
            cluster_obj = KMeans(data = eigvecs,k=self._k,method='kmeans++',max_iter=300,atol=10**-12,rtol=10**-12,verbose=self._verbose)
        else:
            #pass dict arguments 
            self._kmeans_params["data"] = eigvecs
            self._kmeans_params["k"] = self._k
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

#----------------------------
#AffinityPropagation
#----------------------------

class AffinityPropagation(object):

    def __init__(self, data, max_iter = 100,damp=.5 ,similarity_measure='squared_distance', metric='euclidean',
                   sensitivity_weights = 'median',n_break_storage = None ,verbose=True, **kwargs):

        '''
                Affinity Propagation clustering routine
                see http://www.psi.toronto.edu/~psi/pubs2/2007/972.pdf

                Args:
                    data: (n,d)-shaped two-dimensional ndarray or graph adjacency matrix
                    max_iter: maximal iterations count
                    damp: damping parameter for availability and responsability matrix computations in [0,1]
                        A_new = damp*A_new_computed + (1-damp)*A_old
                        NOTE: the choice of the dmaping factor is of particular importance for the outcome of the clustering,
                        numerical oscillations can influence the clustering heavily
                    similarity measure: specification of similarity measure on data array for similarity array generation
                        'distance' for s(x_i,x_j) = -dist(x_i,x_j) w.r.t. to given metric
                        'squared_distance' for s(x_i,x_j) = -dist(x_i,x_j)^2 w.r.t. to given metric
                        None: the given data array will be considered as a similarity array
                            NOTE: it is of crucial importance that s(x_i,x_j)>s(x_i,x_k) => x_i more similar to x_j than to x_k
                            an optimal range for s would be [-inf,0]
                    metric: specification of used metric used in the similarity measures, see scipy.spatial.distance docs
                        WARNING: classic spectral clustering is based on the euclidean distance, it is recommended to use this metric.
                    sensitivity weights: diagonal entries on the similarity matrix.
                        either np.ndarray of shape nxn (larger entries correspond to larger weighting of the point to be chosen as exemplar) or
                        'median': uniform weighting with median sensitivity
                        'min': uniform weighting with low sensitiviy
                        'max': uniform weighting with high sensitivity
                    n_break_storage:
                        None: no break criterion, iteration stops after max_iter. Reasonably faster than setting a break condition
                        int: stops iterations if n_break_storage iterations yield the same exemplars in a row
                    '''

        if type(data) is list:
            raise NotImplementedError('Affinity Propagation is not list compatible yet')
        
        bandwidth = kwargs.get('bandwidth')       

        self._data = data
        self._similarity_measure = similarity_measure
        self._damp = damp
        self._cluster_labels = None
        self._n_clusters = None
        self._verbose = verbose
        self._metric = metric
        self._bandwidth = bandwidth
        self._max_iter = max_iter
        self._sensitivity_weights = sensitivity_weights
        self._n_break_storage = n_break_storage
        
        if self._metric != 'euclidean':
            print('Warning: Affinity Propagation not initialized with euclidean metric. This results in a purely experimental algorithm!')


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
        Runs Affinity Propagation on data.
        '''

        if self._verbose:
           start_time = timer()

        [n_samples,dim] = self._data.shape
        cluster_labels = [None]*n_samples
        
        #----------------------------------------------------
        #similarity matrix
        if self._similarity_measure == 'distance':
                print('Constructing distance matrix')
                S = -distance.pdist(self._data,self._metric)
        if self._similarity_measure == 'squared_distance':
                print('Constructing squared distance matrix')
                S = -np.power(distance.cdist(self._data,self._data,self._metric),2)
        if self._similarity_measure is None:
                S = self._data
                print('Similarity array passed')


        #sensitivity weights
        if self._sensitivity_weights == 'median':
            np.fill_diagonal(S,np.median(S))
        if self._sensitivity_weights == 'min':
            np.fill_diagonal(S,np.min(S))
        if self._sensitivity_weights == 'max':
            np.fill_diagonal(S,np.max(S))
        if type(self._sensitivity_weights) is np.ndarray:
            np.fill_diagonal((S,self._sensitivity_weights))

        counter = 0
        break_cond = False #flags the termination by break condition

        #array allocation
        A = np.zeros((n_samples,n_samples))
        R = np.zeros((n_samples,n_samples))

        if self._n_break_storage is not None:
            break_storage = [0]*self._n_break_storage

        while counter < self._max_iter:
            A_old = np.copy(A)
            R_old = np.copy(R)

            #responsability matrix
            AS = A+S

            max_inds = np.argsort(AS,axis=1)
            maxes = np.max(AS,axis=1)

            maxes_AS = np.tile(maxes[:,np.newaxis],(1,n_samples))
            for i in range(n_samples):
                maxes_AS[i,max_inds[i,-1]] = AS[i,max_inds[i,-2]]
            R = S - maxes_AS
            #damping
            R = self._damp*R + (1-self._damp)*R_old

            #availability matrix
            R_pos = np.maximum(R,0)
            np.fill_diagonal(R_pos,np.diag(R))
            R_sum = np.sum(R_pos,axis=0)
            R_sum_mat = np.tile(R_sum,(n_samples,1))
            R_sum_mat = R_sum_mat - R_pos

            #minimum on offidagonal elements
            M = np.zeros((n_samples,n_samples))
            np.fill_diagonal(M,np.inf)
            A = np.minimum(R_sum_mat,M)

            #damping
            A = self._damp * A + (1 - self._damp) * A_old

            counter = counter+1

            #label assignment in loop if break criterion available
            if self._n_break_storage is not None:
                E = A + R
                indices = np.arange(n_samples)[np.diag(E) > 0]
                # fill break storage
                break_index = counter % self._n_break_storage
                break_storage[break_index] = indices

                # break condition check
                if (counter > self._n_break_storage) and np.all([np.array_equal(x,break_storage[0]) for x in break_storage]):
                    break_cond=True
                    break


        #label assignment after loop if no break criterion available -> saves time
        if self._n_break_storage is None:
            E = A + R
            indices = np.arange(n_samples)[np.diag(E) > 0]

        np.fill_diagonal(S,np.inf)
        cluster_labels = np.argmax(S[:,indices],axis = 1)

        self._cluster_labels = cluster_labels

        if self._verbose:
            if break_cond:
                print('terminated by break condition')
            print('%s iterations until termination.' % str(counter))
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after '+str(elapsed_time))
        #    print('max within-cluster distance to center: %f'%np.max(self._cluster_dist))
        #    print('mean within-cluster distance to center: %f' %np.mean(self._cluster_dist))
        #    print('sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))


#----------------------------
#Hierarchical Clustering
#----------------------------

class HierarchicalClustering(object):

    def __init__(self, data, link='average', num_stop_crit=1, metric='euclidean', verbose=True, **kwargs):
        
        '''
        Hierachical clustering, see https://en.wikipedia.org/wiki/Hierarchical_clustering
        
        TODO:
        
        - avoid deletions of rows in the distance matrix

        Args:
            data: (n,d)-shaped two-dimensional ndarray or distance matrix
            link: specification of the linkage critetion
                'minimum' or 'single' for single linkage (clusters with smallest pointwise minimum distance will be linked)
                'maximum' or 'complete' for complete linkage (clusters with smallest pointwise maximum distance will be linked)
                'average' for average linkage (clusters with smallest pointwise average distance will be linked)
            num_stop_crit=1: number of stopping criteria to be fulfilled to terminate the algorithm: optional arguments 'k' or 'stop_dist' must be passed
                k: number of clusters to produce (will link clusters, until k clusters are remaining)
                stop_dist: stop, when minimum linkage criterion distance between clusters exceeds the given stopping distance stop_dist
            metric='euclidian':
                either: specification of used metric used in the similarity measures, see scipy.spatial.distance docs
                WARNING: classic clustering is based on the euclidean distance, it is recommended to use this metric.
                or: None: the given data array will be considered as the distance matrix already (and not as positions)
            verbose=True:
                whether to print some verbose data after termination.
        '''
        
        k = kwargs.get('k')
        stop_dist = kwargs.get('stop_dist')
        
        self._data = data
        self._link = link
        self._num_stop_crit = num_stop_crit
        self._k = k
        self._stop_dist = stop_dist
        self._cluster_labels = None
        self._verbose = verbose
        self._metric = metric
        
        # Safety checks
        if all((self._metric != 'euclidean', self._metric is not None)):
            print('Warning: hierachical clustering not initialized with euclidean metric or None. This results in a purely experimental algorithm!')
            
        if self._num_stop_crit > 2:
            raise NotImplementedError('currently, just 2 stopping criteria for the algorithm are possible (stop_dist and k)')
        
        # Check, if at least one stopping parameter and also enough stopping parameters are given according to num_stop_crit
        if self._num_stop_crit <= 0:
            raise InvalidValue('number of given stopping criteria must be greater than zero')
        num_given_crit = 0
        if self._k is not None:
            num_given_crit += 1
        if self._stop_dist is not None:
            num_given_crit += 1
        if self._num_stop_crit > num_given_crit:
            raise InvalidValue('more stopping criteria were given than specified stopping parameters')
            
        # Check link type
        assert any((self._link=='average', self._link=='complete', self._link=='single')), "unknown link type"
        
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
        Runs hierarchical clustering on data.
        '''

        if self._verbose:
            start_time = timer()

        [n, dim] = self._data.shape
        cluster_labels = [None] * n
        
        # Init
        cur_num_clusters = n
        clusters = [set([i]) for i in range(n)]
        cluster_dists = distance.squareform(distance.pdist(self._data, self._metric)) if self._metric is not None else self._data
        cluster_sizes = np.ones(n)
        cur_stop_crit = 0
        k_reached = False
        stop_dist_reached = False

        while cur_stop_crit < self._num_stop_crit:

            # Find two clusters to merge, according to crit
            np.fill_diagonal(cluster_dists, cluster_dists.max())
            i_min, j_min = np.unravel_index(cluster_dists.argmin(), (cur_num_clusters, cur_num_clusters))
            ilen = cluster_sizes[i_min]
            jlen = cluster_sizes[j_min]

            # Update cluster distances
            if self._link == 'average':
                cluster_dists[i_min, :] = (ilen * cluster_dists[i_min, :] 
                                           + jlen * cluster_dists[j_min, :]) / (ilen + jlen)
            if self._link == 'single':
                cluster_dists[i_min, :] = np.minimum(cluster_dists[i_min, :], cluster_dists[j_min, :])
            if self._link == 'complete':
                cluster_dists[i_min, :] = np.maximum(cluster_dists[i_min, :], cluster_dists[j_min, :])
            cluster_dists[:, i_min] = cluster_dists[i_min, :]

            # Delete all j_min-Distances
            cluster_dists = np.delete(cluster_dists, j_min, axis=0)
            cluster_dists = np.delete(cluster_dists, j_min, axis=1) # VIELLEICHT IST DAS HIER LANGSAM!!! (vielleicht wäre es besser,
            # diesen index einfach nicht mehr zu benutzen...)

            # Merge the two nearest clusters, adjust sizes
            self._merge_sets(clusters, i_min, j_min)
            cluster_sizes[i_min] += cluster_sizes[j_min]
            cluster_sizes = np.delete(cluster_sizes, j_min)
            cur_num_clusters -= 1
            
            # Count fulfilled stopping criteria
            if self._k is not None:
                if cur_num_clusters <= self._k:
                    k_reached = True
            if self._stop_dist is not None:
                if cluster_dists.min() > self._stop_dist:
                    stop_dist_reached = True
            cur_stop_crit = sum([k_reached, stop_dist_reached])

        # Assign data points to clusters
        self.cluster_labels = np.zeros(n).astype(int)
        for i, cluster in enumerate(clusters):
            for elt in cluster:
                self.cluster_labels[elt] = i

        # Algorithm verbosity
        if self._verbose:
            print("Hierarchical clustering terminated.")
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print("Finished after " + str(elapsed_time))
            if (k_reached):
                print("Stopping cluster number was reached.")
            print("Current number of clusters: {0}".format(cur_num_clusters))
            if (stop_dist_reached):
                print("Stopping distance was reached.")
            print("Current minimum cluster distance: {:.2}".format(cluster_dists.min()))
    
    def _merge_sets(self, X, i, j):
        '''
        Merges two sets of a list of sets.
    
        input:
            X: list of sets
            i: index of the one set to merge. Will also be the merged index
            j: index of the other set to merge. Will be deleted from list
        '''
        X[i] = X[i].union(X[j])
        del X[j]