from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta

from clustertools.models import similarity

#----------------------------
#Consensus Clustering
#----------------------------

class Consensus(object):

    def __init__(self, clusterings, k=5, recluster_what='clusters', how='spectral', spectral_params=None, verbose=True, **kwargs):

        '''
        Consensus clustering, see e.g. http://www.jmlr.org/papers/volume3/strehl02a/strehl02a.pdf
        
        Args:
            cluterings: a list of clustering objects from the clustertools package (or any other objects with a scikit-learn interface, i.p. a cluster_labels-attribute)
            k=5: number of clusterings to produce
            recluter_what='clusters': whether to recluster the jaccard similarity matrix of the clusters and then competing for points (faster) or a hamming similarity matrix of the points (slower)
            how='spectral':
                'spectral': recluster similarity matrix with spectral clustering. Keyword arguments have to be passed:
                    - (e.g. bandwidth for euclidian metric)
                'hierarchical': recluster similarity matrix with hierarchical clustering. Parameters have to or can be to be passed (stopping criteria)
                    - 
            verbose=True:
                whether to print the verbose data of the single algorithms and some data after termination.
            spectral_params: dict containing optional settings for spectral clustering on similarity matrix, see clustertools.models.similarity.SpectralClustering doc. If None, default settings will be applied. Note that "k" and "verbose" parameter of the SpectralClustering instance will always be determined arguments passed to Consensus. Also, "data" will be the similarity matrix and "similarity_measure" will be None.
        '''
        
        self._clusterings = clusterings
        self._k = k
        self._recluster_what = recluster_what
        self._how = how
        self._verbose = verbose
        self._spectral_params = spectral_params
        self._cluster_labels = None
        self._m = len(clusterings)
        
        
        # Safety checks
        assert self._m > 0, "the list of fitted clustering objects is empty"
        self._n = len(clusterings[0].cluster_labels)
        assert any((self._recluster_what == 'clusters', self._recluster_what == 'points')), "unknown self._recluster_what-parameter"
        assert any((self._how == 'spectral', self._how == 'hierarchical')), "unknown how-parameter"
        assert all(tuple([len(clustering.cluster_labels) == self._n for clustering in clusterings])), "clusterings did not cluster same amount of points"
        

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
        Runs consensus clustering on labels of given clustering objects.
        '''

        if self._verbose:
            start_time = timer()

        # Init
        self._cluster_labels = [None] * self._n
        
        # Create array of labels
        labels = np.zeros((self._n, self._m)).astype(int)
        num_unique_labels = np.zeros(self._m).astype(int)
        for j in range(self._m):
            labels[:, j] = self._clusterings[j].cluster_labels
            num_unique_labels[j] = len(np.unique(self._clusterings[j].cluster_labels))
        num_edges = None
        hypergraph = None
        
        # Possibly create hypergraph
        if self._recluster_what == 'clusters':
            num_edges = np.sum(num_unique_labels).astype(int)
            hypergraph = np.zeros((num_edges, self._n)).astype(int)
            cluster_sum = 0
            for j in range(self._m): # for all clusterings
                for l in range(num_unique_labels[j]): # for all cluster labels
                    hypergraph[cluster_sum + l, :] = (labels[:, j] == l)
                cluster_sum += num_unique_labels[j]
        
        # Calculate distance matrix of hyperedges or points
        if self._recluster_what == 'clusters':
            dists = distance.squareform(distance.pdist(hypergraph, 'jaccard'))
        elif self._recluster_what == 'points':
            dists = distance.squareform(distance.pdist(labels, 'hamming'))
        
        # Recluster
        meta_clustering = None
        if self._how == 'hierarchical':
            meta_clustering = similarity.HierarchicalClustering(dists, metric=None, k=self._k, verbose=self._verbose)
        if self._how == 'spectral':
            if self._spectral_params is None:
                # Default spectral clustering
                meta_clustering = similarity.SpectralClustering(dists, similarity_measure=None, n=self._k, verbose=self._verbose)
            else:
                self._spectral_params["data"] = dists
                self._spectral_params["similarity_measure"] = None
                self._spectral_params["k"] = self._k
                self._spectral_params["verbose"] = self._verbose
                meta_clustering = similarity.SpectralClustering(**self._spectral_params)
        meta_clustering.fit()
        meta_labels = meta_clustering.cluster_labels
        
        # Possibly for points
        if self._recluster_what == 'points':
            self._cluster_labels = meta_labels
        if self._recluster_what == 'clusters':
            point_label_buckets = np.zeros((self._n, self._k)).astype(int)
            for edge_ind in range(num_edges):
                point_label_buckets[:, meta_labels[edge_ind]] += hypergraph[edge_ind]
            for i in range(self._n):
                self._cluster_labels[i] = np.random.choice(np.flatnonzero(point_label_buckets[i, :] == point_label_buckets[i, :].max()))

        # Algorithm verbosity
        if self._verbose:
            print("Consensus clustering terminated.")
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print("Finished after " + str(elapsed_time))