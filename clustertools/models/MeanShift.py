from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.spatial import distance
from timeit import default_timer as timer
from datetime import timedelta
from scipy.stats import gaussian_kde
__metaclass__ = type


class MeanShift(object):
    """class provides mean shift clustering for n data points in d dimensions.
    
       references: 
           https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
           Mean Shift: A Robust approach towards feature space analysis - Comaniciu, Meer
    """
    
    def __init__(self, data, bandwidth='scott', max_iter=150, kernel='gaussian', metric = 'euclidean', atol=1e-03, verbose=True):
        '''
        Args:
            data: (n,d)-shaped d-dimensional ndarray objects containing float/integer data 
            bandwidth: affects how many clusters are formed, generally a bigger 
                        bandwidth -> less, broader clusters and smaller bandwidth -> more clusters, 
                        extreme case: each data point is its own cluster
                        either give a scalar for the bandwidth, or the bandwidth is estimated using 
                        the 'scott' or 'silverman' rule
            max_iter: int, maximal iterations before terminating
            kernel: typically a Gaussian kernel is used, so far no other kernels implemented
            metric: metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.pdist
            atol: absolute threshold for point to stop shifting. 
        '''
        self._bandwidth = bandwidth 
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
        self._outliers = False

        if self._metric != 'euclidean':
            print('Initialized with %s metric. Use euclidean metric for classic Mean shift algorithm. \n'
                  'Bad things might happen, depending on your dataset and used metric.'%metric)

    def fit(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.        
        '''
        
        if self._verbose:
            start_time = timer()
        
        self.determine_bandwidth() #if not given a bandwidth, use rule to estimate
        
        shifted = np.copy(self._data)
        [n,d] = np.shape(self._data) #number of data points and dimension
        
        stillshifting = [True]*n #stop shifting points when the difference in position < absolute tolerance           
        counter = 0 
    
        while counter < self._max_iter and stillshifting!=[False]*n:
            for i in range(0,n):    
                 if stillshifting[i] == True:
                     kerneldata =  self.gaussianKernel(self._data, shifted[i])
                     shift = np.sum(self._data*kerneldata,0)/np.sum(kerneldata) 
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
            print('There are %f outliers' %self._outliers)
            print('max within cluster distance to center: %f'%np.max(self._cluster_dist))
            print('mean within cluster distance to center: %f' %np.mean(self._cluster_dist))
            print('sum of within cluster squared errors: %f' % np.sum(np.square(self._cluster_dist)))
    
    def determine_bandwidth(self):
        '''
        If not bandwidth (scalar) is given, determine using either scott's or silverman's rule
        D.W. Scott, “Multivariate Density Estimation: Theory, Practice, and Visualization”, John Wiley & Sons, New York, Chicester, 1992
        W. Silverman, “Density Estimation for Statistics and Data Analysis”, Vol. 26, Monographs on Statistics and Applied Probability, Chapman and Hall, London, 1986.
        '''
        [n,d] = np.shape(self._data)
        if isinstance(self._bandwidth,int) or isinstance(self._bandwidth,float):
            pass
        elif self._bandwidth == 'scott':
            self._bandwidth = n**(-1./(d + 4))
        elif self._bandwidth == 'silverman':
            self._bandwidth = (n*(d + 2)/4.)**(-1./(d + 4))
        else:
            print("The bandwidth has to be either a real number or has to be estimated using 'scott' or 'silverman'")
    
    
    def clusterCenters(self):
        '''
        Given the shifted points, group them to clusters, assigning labels 1,..., # of clusters; 
        the center of a cluster corresponds to the local maxima of the point density; 
        the distance to cluster centers is calculated using the chosen metric. 
        Outliers are also detected and assigned the label 0, an outlier here is 
        defined as a cluster with just one member.
        
        (inspired by regular space clustering: http://docs.markovmodel.org/lecture_regspace.html)
        '''
        n, d = int(np.shape(self._results)[0]), int(np.shape(self._results)[1])      
        labels = np.zeros(n)
        count = 0 #count cluster centers
        labels[0] = 0 #first point is assigned to 0, outliner, until another point close by is found
        for k in np.arange(1,n):
            dist = distance.cdist(self._results[:k,:], [self._results[k,:]], metric =self._metric)
            index_min = np.argmin(dist)
            minimum = np.min(dist)
            if minimum< self._mindist:
                if labels[index_min] == 0:
                    count +=1
                    labels[k] = count
                    labels[index_min] = count#
                else:
                    labels[k] = labels[index_min]
            else:
                labels[k] = 0
                
        if np.shape(self._results[labels==0])[0]==0:
            clustercenters = np.array([np.mean(self._results[labels==k],axis=0) for k in np.arange(1, count+1)]) 
            index = labels-1
            self._outliers = 0
        else:
            clustercenters = np.array([np.mean(self._results[labels==k],axis=0) for k in np.arange(0, count+1)]) 
            index = labels
            self._outliers = np.shape(self._results[labels==0])[0]
        
        index = index.astype(int)
        diffvector = self._results - clustercenters[index]
        clusterdist = distance.cdist(diffvector, np.array([[0]*d]), metric=self._metric)
        self._cluster_centers = clustercenters
        self._cluster_dist = clusterdist
        self._cluster_labels = labels

    def gaussianKernel(self, points, point):
        '''
        Calculates the Gaussian Kernel of points[i]-point for each element i in the points array; 
        the value of the kernel depends strongly on the chosen bandwidth and also on the distance metric
        '''
        distancek = distance.cdist(points,[point], metric=self._metric)
        kernel = 1/(self._bandwidth*np.sqrt(2*np.pi)) * np.exp(- distancek**2 /(2* self._bandwidth**2))
        return kernel

def plot_kde(xpts, ypts, bandwidth):
    '''
    Kernel density estimation of xpoints, ypoints (ie in 2D), returns X,Y,Z for plotting
    '''
    xmin, xmax = xpts.min(), xpts.max()
    ymin, ymax = ypts.min(), ypts.max() 
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([xpts, ypts])
    kernel = gaussian_kde(values,bw_method=bandwidth/np.asarray(np.vstack([xpts, ypts]).std(ddof=1))) #need to rescale bandwidth to match
    Z = np.reshape(kernel(positions).T, X.shape)
    return [X, Y, Z]

