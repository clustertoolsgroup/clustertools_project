from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta

class HierarchicalClustering(object):

    def __init__(self, data, link='average', num_stop_crit=1, metric='euclidean', verbose=True, **kwargs):
        
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
            if all((self._k is not None, cur_num_clusters <= self._k)):
                k_reached = True
            if all((self._stop_dist is not None, cluster_dists.min() > self._stop_dist)):
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

    def some_method(self,k,distances,mode='mutual'):
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
        W = np.zeros(k)
        return W
