# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:36:19 2017

@author: luzie
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta
from scipy.stats import gaussian_kde
__metaclass__ = type


class MeanShift(object):
    """class provides mean shift clustering for n data points in d dimensions.
    
       references: 
           https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
           "Mean Shift: A Robust approach towards feature space analysis" Comaniciu, Meer"""
    
    def __init__(self, data, bandwidth, max_iter=150, kernel='gaussian', metric = 'euclidean', atol=1e-03, verbose=True):
        '''
        Args:
            data: (n,d)-shaped 2-dimensional ndarray objects containing float data or a list consisting of
            fitting ndarrays
            bandwidth: affects how many clusters are formed, generally
                        bigger bandwidth-less clusters, maybe just one and 
                        smaller bandwidth - more clusters, extreme case: each data point is its one cluster
            max_iter: int, maximal iterations before terminating
            kernel: typically a Gaussian kernel is used
            metric: metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist
            atol: absolute threshold for point to stop shifting. 
        '''
        self._bandwidth = bandwidth #to do: bandwidth , eg scotts rule, silvermans rule
        self._max_iter = max_iter
        self._data = data
        self._kernel = kernel
        self._metric = metric
        self._atol = atol
        self._mindist = None
        self._cluster_centers = None
        self._results = None
        self._cluster_labels = None
        self._cluster_dist = None
        self._verbose = verbose
        self._fitted = False

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)

    def fit(self):
        '''
        CHANGE TEXT...
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
        
        if self._verbose:
            start_time = timer()

        
        shifted = np.copy(self._data)
        [n,d] = np.shape(self._data) #number of data points and dimension
        
        stillshifting = [True]*n #stop shifting points when the difference in position < absolute tolerance           
        counter = 0 #count iterations
    
        while counter < self._max_iter and stillshifting!=[False]*n:
            for i in range(0,n):    
                 if stillshifting[i] == True:
                     kerneldata =  gaussianKernel(self._data, shifted[i], self._bandwidth, self._metric)
                     shift = np.sum(self._data*kerneldata,0)/np.sum(kerneldata) #to do: check for non zero denominator
                     if distance.pdist(np.array([shift, shifted[i]]), metric =self._metric)[0] < self._atol:
                         stillshifting[i] = False
                     shifted[i] = shift
            counter += 1

        self._results = shifted
        self._mindist = np.mean([max(self._results[:,k])-min(self._results[:,k]) for k in range(d)])/np.power(n,1/d)
        self.clusterCenters()

        self._fitted = True
        
        
        if self._verbose:

            print('%s iterations until termination.' % str(counter))
            print('used bandwidth: %f' % self._bandwidth)
            elapsed_time = timer() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            print('Finished after '+str(elapsed_time))
            print('number of cluster found: %f'% np.max(self._cluster_labels))
            print('max within cluster distance to center: %f'%np.max(self._cluster_dist))
            print('mean within cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))
    
    def clusterCenters(self):
        '''
        Given the shifted points, group them to clusters, assigning labels 1,..., number of clusters; 
        the center of a cluster corresponds to the local maxima of the point density; 
        the distance to cluster centers is calculated using the chosen metric
        
        (inspired by regular space clustering: http://docs.markovmodel.org/lecture_regspace.html)
        '''
        n, d = int(np.shape(self._results)[0]), int(np.shape(self._results)[1])      
        labels = np.zeros(n)
        count = 1 #count cluster centers
        labels[0] = count #first point is assigned to center 1
        for k in np.arange(1,n):
            dist = distance.cdist(self._results[:k,:], [self._results[k,:]], metric =self._metric)
            index_min = np.argmin(dist)
            minimum = min(dist)
            if minimum< self._mindist:
                labels[k] = labels[index_min]
            else:
                count +=1
                labels[k] = count
        clustercenters = np.array([np.mean(self._results[labels==k],axis=0) for k in np.arange(1, count+1)])              
        index = labels-1
        index = index.astype(int)
        diffvector = self._results - clustercenters[index]
        clusterdist = distance.cdist(diffvector, np.zeros((1,d)), metric=self._metric)
        self._cluster_labels = labels
        self._cluster_dist = clusterdist
        self._cluster_centers = clustercenters
        #to do: classify outliers, maybe label 0 or -1, (ie clusters with only one point)
 

def gaussianKernel(points, point, bandwidth, chosenmetric):
    '''
    '''
    distancek = distance.cdist(points,[point], metric=chosenmetric)
    kernel = 1/(bandwidth*np.sqrt(2*np.pi)) * np.exp(- distancek**2 /(2* bandwidth**2))
    return kernel

def plot_kde(xpts, ypts, bandwidth):
    '''
    '''
    xmin, xmax = xpts.min(), xpts.max()
    ymin, ymax = ypts.min(), ypts.max() 
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xpts, ypts])
    kernel = gaussian_kde(values,bw_method=bandwidth)
    Z = np.reshape(kernel(positions).T, X.shape)
    return [X, Y, Z]

