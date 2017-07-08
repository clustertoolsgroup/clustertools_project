from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer

class FuzzyCMeans(object):
    ''' class provides Fuzzy C Means Algorithm for clustering n data points in d dimensions, 
    (depending on the number of clusters). 
    
    references: 
        Bezdek, Ehrlich, and Full. "FCM: The fuzzy c-means clustering algorithm." 1984.
    '''

    def __init__(self,data,c,m = 2, epsilon=1e-02, maxiter=30, metric='euclidean',verbose=True):
        '''
        Args:
            data: (n,d)-shaped d-dimensional ndarray objects containing float/integer data to be clustered
            c: integer, number of clusters 2<=c<=number of data points
            m: weight exponent 1<=m, large m -> fuzzier clustering, m=1 -> crisp partitioning
            epsilon: small, positive value, if norm of difference of successive membership matrices is smaller than epsilon, the iteration stops
            maxiter: integer, maximum number of iterations
            metric:  metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist       
        '''

        if type(data) is list:
            raise NotImplementedError('FuzzyCMeans is not list compatible yet')

        self._data = data
        self._c = c
        self._m = m 
        self._eps = epsilon
        self._maxiter = maxiter
        self._cluster_labels = None
        self._cluster_centers = None
        self._cluster_dist = None
        self._metric = metric
        self._verbose = verbose
        self._membership = None
        self._iter = None
        self._time = None

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)

    def fit(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.     
        '''

        start_time = timer()
        
        #initialize
        [n,d] = self._data.shape
        #initial membership matrix
        randommatrix = np.random.random((self._c, n)) 
        #normalize s.t. column sum = 1
        Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0))) 
        #row sum >0 for all rows
        while (np.sum(Uk, axis=1)<=0).any(): 
            randommatrix = np.random.random((self._c, n))
            Uk = np.dot(randommatrix,np.diag(1/np.sum(randommatrix, axis=0)))
        
        #iterate  
        for k in range(self._maxiter):
            Uk = np.dot(Uk,np.diag(1/np.sum(Uk, axis=0)))
            Uk_powerm = np.power(Uk,self._m)
            #compute cluster centers
            vk = np.dot(np.diag(1/np.sum(Uk_powerm,axis=1)),np.dot(Uk_powerm, self._data))
            #distance matrix
            D = distance.cdist(vk, self._data, metric = self._metric) 
            Ukplus1 = np.power(D,- 2. / (self._m - 1))
            Ukplus1 /=np.dot(np.ones((self._c, 1)),[Ukplus1.sum(axis=0)])
        
            if np.linalg.norm(Uk-Ukplus1) <self._eps:
                break
            Uk = Ukplus1.copy()
                
            
        self._cluster_labels = np.argmax(Uk,axis=0)
        self._membership = Uk
        self._cluster_centers = vk
        self._cluster_dist = np.min(D,axis=0)
        self._iter = k
        elapsed_time = timer() - start_time
        self._time = elapsed_time
        if self._verbose: 
            print('Finished after ' + str(elapsed_time))
            print('%s iterations until termination.' % str(k))
            print('Max within cluster distance to center: %f'%np.max(self._cluster_dist))
            print('Mean within cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('Sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))
