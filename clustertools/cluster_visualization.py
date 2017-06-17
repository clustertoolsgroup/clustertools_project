from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles

from mcmm import clustering as cl
from mcmm import DBSCAN

class ClusterViz(object):
    '''
    Class serving as a wrapper for matplotlib corresponding with mcmm.clustering classes to provide
    visualiziation defaults.
    '''

    def __init__(self,cluster_object):
        self._cluster_object = cluster_object

    def scatter(self,feature_indices=None,mark_centers = True,color_clusters=False,sample_rate=20):
        '''
        Produces a scatter plot of the data stored in the cluster object.
        Args:
            feature_indices: indices of the features which should be plotted.
                Possible Arguments:
                    A list containing 2 or 3 indices denoting the feature indices in the data which should
                    be incorporated to the plot,
                    None. Falls back to plotting all features of the data and produces an error if number of features is
                    not 2 or 3.
            mark_centers:
            color_clusters:
            sample_rate: int. Steps in which the underlying data is sampled, i.e. "20" corresponds to plotting of every
            20th observation from the data

        NOTE: if the clustering instance is not fitted, ClusterViz.scatter will fit the object to its passed data
        '''

        if not self._cluster_object.fitted:
            self._cluster_object.fit()

        color = 'grey'
        if feature_indices is None:
            n_features = self._cluster_object.data.shape[1]
            feature_indices = range(self._cluster_object.data.shape[1])
        elif type(feature_indices) is list:
            n_features = len(feature_indices)
        else:
            raise TypeError('feature_indices needs to be a list or None')

        if color_clusters:
            if type(self._cluster_object.cluster_labels) is list:
                color = np.concatenate(self._cluster_object.cluster_labels,axis=0)
            else:
                color = self._cluster_object.cluster_labels
            color = color[::sample_rate]
        if n_features == 2:
            fig = plt.figure()
            x = self._cluster_object.data[::sample_rate,feature_indices[0]]
            y = self._cluster_object.data[::sample_rate,feature_indices[1]]
            plt.scatter(x,y,c=color)
            if mark_centers:
                x = self._cluster_object.cluster_centers[:,feature_indices[0]]
                y = self._cluster_object.cluster_centers[:,feature_indices[1]]
                plt.scatter(x,y,c='r',s=50)
        elif n_features == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = self._cluster_object.data[::sample_rate,feature_indices[0]]
            y = self._cluster_object.data[::sample_rate,feature_indices[1]]
            z = self._cluster_object.data[::sample_rate,feature_indices[2]]
            ax.scatter(x,y,z,c=color)
            if mark_centers:
                x = self._cluster_object.cluster_centers[:,feature_indices[0]]
                y = self._cluster_object.cluster_centers[:,feature_indices[1]]
                z = self._cluster_object.cluster_centers[:,feature_indices[2]]
                ax.scatter(x,y,z,c='r',s=50)

        else:
            raise ValueError('Not able to plot with given feature_indices')
        plt.show()

    def elbow(self,n_clusters):
        '''
        Implemetation of the elbow-rule plot (within cluster SSE vs. number of clusters), which is especially useful
        for KMeans fitting evaluation for shorter fitting times and/or smaller k
        Args:
            n_clusters: list of integers specifying the number of clusters

        NOTE: the used cluster instance has to support the parameter 'k' in its .fit method. Using KMeans will work,
        using for example Regspace will not for obvious reasons.
        '''

        SSE_list = []
        for k in n_clusters:
            try:
                self._cluster_object.fit(k,verbose=False)
            except:
                ValueError('clustering instance must support number of cluster centers as parameter')
            if type(self._cluster_object.cluster_dist) is list:
                cluster_dist = np.concatenate(self._cluster_object.cluster_dist,axis=0)
            else:
                cluster_dist = self._cluster_object.cluster_dist
            SSE = np.sum(np.square(cluster_dist))
            SSE_list.append(SSE)

        fig = plt.figure()
        plt.plot(n_clusters,SSE_list)
        plt.suptitle('within-cluster SSE vs. k')
        plt.xlabel('$k$')
        plt.ylabel('$SSE$')
        plt.show()

#-------------------------------
# cluster test visualizations
#-------------------------------
def kmeans_blobs_2d(n_samples,n_clusters,k,method='kmeans++',std=1):
    '''
    generates random dataset by sklearn.datasets.samplesgenerator.make_blobs
    and visualizes the mcmm.analysis.KMeans clustering algorithm via pyplot

        Args:
        n_samples: number of observations in dataset
        n_clusters: number of clusters in dataset
        k: number of cluster centers to be determined by k-means
        method: the KMeans method, i.e. 'forgy' or 'kmeans++'
        std: the cluster intern standard deviation of the generated dataset
    '''

    data = make_blobs(n_samples,2,n_clusters,cluster_std=std)[0]
    kmeans = cl.KMeans(data,k,method)
    cluster_centers = kmeans.cluster_centers
    cluster_labels = kmeans.cluster_labels

    plt.scatter(data[:, 0], data[:, 1],c=cluster_labels)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', s=50)
    plt.show()


def kmeans_blobs_3d(n_samples,n_clusters,k,method='kmeans++',std=1):
    '''
    generates random dataset by sklearn.datasets.samplesgenerator.make_blobs
    and visualizes the mcmm.analysis.KMeans clustering algorithm via pyplot

        Args:
        n_samples: number of observations in dataset
        n_clusters: number of clusters in dataset
        k: number of cluster centers to be determined by k-means
        method: the KMeans method, i.e. 'forgy' or 'kmeans++'
        std: the cluster intern standard deviation of the generated dataset
    '''

    data = make_blobs(n_samples,3,n_clusters,cluster_std=std)[0]
    kmeans = cl.KMeans(data,k,method)
    cluster_centers = kmeans.cluster_centers
    cluster_labels = kmeans.cluster_labels

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1],data[:,2],c=cluster_labels)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],cluster_centers[:,2], c='r', s=150,depthshade=False)
    plt.show()

def DBSCAN_cirles(n_samples=10000,factor=.4,noise=.1,eps=.1,minPts=30):
    '''
    Plots classic example for DBSCAN clustering on datasets consisting of two circular clusters
    Args:
        n_samples: number of total observations
        factor: scaling between inner and outer circle, see sklearn.datasets.make_circles doc
        noise: standard deviation of noise, see sklearn.datasets.make_circles doc
        eps: DBSCAN epsilon parameter
        minPts: DBSCAN minPts parameter
    '''
    circle = make_circles(n_samples=n_samples, factor=factor, noise=noise)
    circle = circle[0]
    circlescan = DBSCAN.DBSCAN(circle,eps,minPts)
    #reassign noise for plotting
    labels = circlescan.cluster_labels
    for p, i in enumerate(labels):
        if i == 'noise':
            labels[p] = circlescan._n_clusters + 1
    plt.scatter(circle[:, 0], circle[:, 1], c=circlescan.cluster_labels)
