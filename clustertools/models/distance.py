from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
from random import sample
import numpy as np
from scipy.spatial import distance
from scipy.stats import rv_discrete
from timeit import default_timer as timer
from datetime import timedelta

#----------------
#K-Means clustering
#----------------

class KMeans(object):
    '''
    Class providing simple k-Means clustering for (n,d)-shaped trajectory ndarray objects containing float data
    or list of trajectory ndarrays each with fitting second dimension d
    '''

    def __init__(self,data,k,max_iter=150,method="forgy",metric='euclidean',atol=1e-03,rtol=1e-03,verbose=True):
        '''
        Args:
            data: (n,d)-shaped 2-dimensional ndarray objects containing float data or a list consisting of
            fitting ndarrays
            k: int, number of cluster centers. required to be <= n.
            max_iter: int, maximal iterations before terminating
            method: way of initializing cluster centers. Use 'forgy' for Forgys method or 'kmeans++'
            metric: metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.cdist
            atol,rtol: absolute and relative tolerance threshold to stop iteration before reaching max_iter. see numpy.allclose documentation
        '''
        self._k = k
        self._max_iter = max_iter
        self._data = data
        self._method = method
        self._metric = metric
        self._rtol = rtol
        self._atol = atol
        self._cluster_centers = None
        self._cluster_labels = None
        self._cluster_dist = None
        self._traj_list_indices = None
        self._verbose = verbose
        self._fitted = False
        self._data_type_list = None

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic KMeans. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)

    @property
    def cluster_centers(self):
        if self._cluster_centers is None:
            self.fit()
        return self._cluster_centers
    @cluster_centers.setter
    def cluster_centers(self,value):
        self._cluster_centers = value

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            self.fit()
        return self._cluster_labels
    @cluster_labels.setter
    def cluster_labels(self,value):
        self._cluster_labels = value

    @property
    def cluster_dist(self):
        if self._cluster_dist is None:
            self.fit()
        return self._cluster_dist
    @cluster_dist.setter
    def cluster_dist(self,value):
        self._cluster_dist = value

    @property
    def fitted(self):
        return self._fitted

    @property
    def data(self):
        return self._data


    def fit(self,k=None,verbose=None):
        '''
        Runs the clustering iteration on the data it was given when initialized. If the object is not fitted,
        accessing .cluster_labels, .cluster_centers and .cluster_dist will also lead to a call of .fit().

        You can specify k and verbose parameters ether in the .fit method or when initializing the KMeans instance.s

        Cluster centers,cluster labels and distances to associated center for the given data will
        be stored in the objects properties. An initial list of data will return labels and
        distances in lists accordingly to match the initial data list.

        NOTE: multiple calls of .fit() are possible and will yield different outcomes, since KMeans
        always has a random component due to its cluster initialization. Just accessing the properties
        .cluster_labels, .cluster_centers and .cluster_dist will however NOT change the stored properties.
        '''
        if k is not None:
            self._k = k
        if verbose is not None:
            self._verbose = verbose
        if self._verbose:
            start_time = timer()
        if self._data_type_list is None:
            self._data_type_list = type(self._data) is list
        if self._data_type_list and not self._fitted:
            self._data, self._traj_list_indices = concat_list(self._data)
        cluster_centers = initialize_centers(self._data, self._k, self._method)

        counter = 0
        break_cond = False # flags the termination by break condition

        while counter < self._max_iter:
            cluster_labels, cluster_dist = get_cluster_info(self._data, cluster_centers, metric=self._metric)
            new_cluster_centers = set_new_cluster_centers(self._data, cluster_labels, self._k)
            #break condition
            if np.allclose(cluster_centers, new_cluster_centers, self._atol, self._rtol):
                break_cond = True
                cluster_centers = new_cluster_centers
                break
            cluster_centers = new_cluster_centers
            counter = counter+1

        cluster_labels, cluster_dist = get_cluster_info(self._data, cluster_centers, metric=self._metric)
        self._cluster_centers = cluster_centers
        #cutting of labels according to given list
        if self._data_type_list:
            cluster_labels=np.split(cluster_labels,self._traj_list_indices[:-1])
            cluster_dist = np.split(cluster_dist,self._traj_list_indices[:-1])
        self._cluster_labels = cluster_labels
        self._cluster_dist = cluster_dist

        self._fitted = True
        if self._verbose:
            if break_cond:
                print('terminated by break condition')
            print('%s iterations until termination.' % str(counter))
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after '+str(elapsed_time))
            print('max within-cluster distance to center: %f'%np.max(self._cluster_dist))
            print('mean within-cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))


    def transform(self,data):
        '''
        Returns cluster labeling for additional data corresponding
        to existing cluster centers stored in the object. (Also fits to initial data, if not fitted before)
        Args:
            data: (n,d)-shaped ndarray or list consisting of ndarrays each with matching second dimension d
        Returns:
            cluster labels for passed data argument and cluster distances with respect to the given metric
        '''
        array_type = type(data)
        if array_type is list:
            data, traj_list_indices = concat_list(data)

        if not self._fitted:
            self.fit()

        cluster_labels, cluster_dist = get_cluster_info(data, self.cluster_centers, metric=self._metric)

        if array_type is list:
            cluster_labels = np.split(cluster_labels, traj_list_indices[:-1])
            cluster_dist = np.split(cluster_dist, traj_list_indices[:-1])

        self._fitted = True
        return cluster_labels, cluster_dist



#-------------------
#Regspace clustering
#-------------------

class Regspace(object):
    '''Regular space clustering.'''

    def __init__(self,data,max_centers,min_dist,metric='euclidean',verbose=True):
        '''

        Args:
            data: ndarray containing (n,d)-shaped float data or list of arrays each with coninciding second
            dimension
            max_centers: the maximal cluster centers to be determined by the algorithm before stopping iteration,
            integer greater than 0 required
            min_dist: the minimal distances between cluster centers
            metric: the metric used to determine distances d-dimensional space. Default = euclidean.
            See scipy.spatial.distance.cdist for possible metrics
        '''

        self._data = data
        self._max_centers = max_centers
        self._min_dist = min_dist
        self._metric = metric
        self._cluster_centers = None
        self._cluster_labels = None
        self._cluster_dist = None
        self._traj_list_indices = None
        self._verbose = verbose
        self._fitted = False
        self._data_type_list = None

    @property
    def cluster_centers(self):
        if self._cluster_centers is None:
            self.fit()
        return self._cluster_centers

    @cluster_centers.setter
    def cluster_centers(self, value):
        self._cluster_centers = value

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            self.fit()
        return self._cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, value):
        self._cluster_labels = value

    @property
    def cluster_dist(self):
        if self._cluster_dist is None:
            self.fit()
        return self._cluster_dist

    @cluster_dist.setter
    def cluster_dist(self, value):
        self._cluster_dist = value

    @property
    def fitted(self):
        return self._fitted
    @property
    def data(self):
        return self._data


    def fit(self):
        '''
        performs regspace clustering on the data and provides cluster centers, clusterlabels and cluster distances
        '''
        if self._verbose:
            start_time = timer()

        if self._data_type_list is None:
            self._data_type_list = type(self._data) is list

        if self._data_type_list and not self._fitted:
            self._data,self._traj_list_indices = concat_list(self._data)
        center_list = [self._data[0, :]]
        num_observations,d = self._data.shape

        for i in range(1,num_observations):
            if len(center_list) >= self._max_centers:
                break
            x_active = self._data[i, :]
            distances = distance.cdist(x_active.reshape(1,d), np.array(center_list).reshape(len(center_list),d), metric=self._metric)
            if np.all(distances > self._min_dist):
                center_list.append(x_active)

        cluster_centers = np.array(center_list)
        cluster_labels, cluster_dist = get_cluster_info(self._data, cluster_centers, self._metric)
        self.cluster_centers = cluster_centers
        if self._data_type_list:
            cluster_labels = np.split(cluster_labels, self._traj_list_indices[:-1])
            cluster_dist = np.split(cluster_dist, self._traj_list_indices[:-1])
        self._cluster_labels = cluster_labels
        self._cluster_dist = cluster_dist
        self._fitted = True

        if self._verbose:
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after '+str(elapsed_time))
            print('%i cluster centers detected'%len(self._cluster_centers)+'\n')
            print('max within-cluster distance to center: %f'%np.max(self._cluster_dist))
            print('mean within-cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('within cluster sum of squared errors: %f' % np.sum(np.square(self._cluster_dist)))

    def transform(self,data):
        raise NotImplementedError


#--------------
#global functions
#--------------


def concat_list(array_list):
    '''
    for a given list of ndarrays, concatenate to a single numpy array
    and also return number of observations in each array to make reshaping of
    cluster labeling according to passed lists possible
    '''

    traj_list_indices = [array.shape[0] for array in array_list]
    traj_list_indices = np.cumsum(traj_list_indices)
    return np.concatenate(array_list,axis=0), traj_list_indices

def get_cluster_info(data,cluster_centers,metric='euclidean'):
    '''
    For (n,d)-shaped float data and given centroids, returns the corresponding cluster centers and corresponding labeling
    with respect to a metric.

    Args:
        data: (n,d) ndarray
        cluster_centers: (k,d) ndarray
        metric: metric parameters used as in scipy.spatial.distance.cdist. uses euclidean metric as default.

    Returns:
        cluster_labels: (d,1) vector containing the corresponding cluster centers of each of the data rows
        cluster_dist (d,1) vector containing squared distance of data observation to corresponding cluster centroid
    '''

    distance_matrix = distance.cdist(data,cluster_centers,metric)
    cluster_labels = np.argmin(distance_matrix,axis=1)
    cluster_dist = np.min(distance_matrix,axis=1)
    return cluster_labels, cluster_dist


def optimize_centroid(cluster_points):
    '''
    for a given set of observations in one cluster, compute and return a new centroid
    '''
    vecsum = np.sum(cluster_points,axis=0)
    centroid = np.divide(vecsum,cluster_points.shape[0])
    return centroid

def set_new_cluster_centers(data,cluster_labels,k):
    '''
    for given data and clusterlabeling, construct new centers for each cluster
    '''
    center_list = []
    for i in range(k):
        new_center = optimize_centroid(data[cluster_labels == i,:])
        center_list.append(new_center)

    return np.vstack(center_list)

def initialize_centers(data,k,method):
    '''
    initializes cluster centers with respect to given method
    '''

    if method == 'forgy':
        cluster_centers = forgy_centers(data,k)
    elif method == 'kmeans++':
        cluster_centers = kmeans_plusplus_centers(data,k)
    return cluster_centers

#---------
#cluster center initializations
#---------

def forgy_centers(data,k):
    '''
    returns k randomly chosen cluster centers from data
    '''
    return sample(list(data),k)


def kmeans_plusplus_centers(data,k):
    '''
    returns cluster centers initialized by kmeans++ method,
    see http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    '''

    index_vals = range(data.shape[0])

    c1 = sample(list(data),1)
    center_list = np.array(c1)
    if k == 1:
        return c1
    while center_list.shape[0]<k:
        labels,distances = get_cluster_info(data,center_list)
        D2 = D2_weighting(distances)
        distribution = rv_discrete(values=(index_vals,D2))
        center_choice = np.array(data[distribution.rvs(size=1),:])
        center_list = np.vstack([center_list,center_choice])
    return np.array(center_list)


def D2_weighting(dist_array):
    '''
    performs the D^2-probability weighting on an ndarray of cluster distances associated to data points,
    see http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    returns kmeans++ probability distribution vector
    '''
    D2 = np.square(dist_array)
    sum = np.sum(D2)
    D2 = np.divide(D2,sum)
    return D2




