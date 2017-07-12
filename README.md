
## Clustertools

### Overview

A self-contained, scikit-learn API-oriented Python package supporting a variety of different clustering algorithms for Machine Learning applications.

Developed as part of "Mathematical aspects in Machine Learning", Freie Universität Berlin, summer semester 2017.

![](https://raw.githubusercontent.com/clustertoolsgroup/clustertools_project/master/images/meanshift_examples.png)

### Contents
This package provides the following functionality:

**Distance-based clustering**:

- Regular space clustering 
- K-Means/K-Means++ (Lloyd-type iterative method)

**Density-based clustering**:

- DBSCAN
- Mean shift

**Similarity/graph-based clustering**:

- Spectral clustering (Normalized cuts method/standardized graph Laplacian)
- Affinity Propagation
- Hierarchical clustering

Note: Spectral clustering and Affinity propagation offer options to cluster on abstract graphs and adjacency/similarity arrays.

**Fuzzy methods**:

- Fuzzy C-Means

**Consensus clustering (similarity-based)**:

- By reclustering points
- By reclustering clusters and competing for points

### Example

```python
from clustertools.models.density import MeanShift
from clustertools.load_data import load_spiral_data

data = load_spiral_data(scale=True)

ms_instance = MeanShift(data,**params)
ms_instance.fit()
```

### Requirements:
numpy, scipy, matplotlib

### Installation
Call
```python setup.py install```
from project directory. Alternatively, call
```
pip install git+https://github.com/clustertoolsgroup/clustertools_project
``` 
to install from github.

### References

Arthur, David and Sergei Vassilvitskii. "How Slow is the K-means Method?". Proceedings of the Twenty-second Annual Symposium on Computational Geometry. SCG '06. New York, NY, USA: ACM: 144–153 (2006).

Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful seeding." Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035 (2007).

Bezdek, James C., Robert Ehrlich, and William Full. "FCM: The fuzzy c-means clustering algorithm." Computers \& Geosciences 10.2-3 (1984): 191-203

Comaniciu, Dorin, and Peter Meer. "Mean shift: A robust approach toward feature space analysis." IEEE Transactions on pattern analysis and machine intelligence 24.5 (2002): 603-619.

Dhillon, Inderjit S. and Guan, Yuqiang and Kulis, Brian.
"Kernel K-means: Spectral Clustering and Normalized Cuts."
Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2004): 551--556

Ester, Martin and Hans-Peter Kriegel and Jörg Sander and Xiaowei Xu.
"A density-based algorithm for discovering clusters in large spatial databases with noise." Proceedings of the Second International Conference on Knowledge Discovery and Data Mining
(1996): 226--231
   
Fränti, Pasi. "Clustering datasets." Clustering datasets. N.p., 2015. Web. 07 July 2017. <http://cs.uef.fi/sipu/datasets/>.
   
Frey, Brendan J. and Delbert Dueck. "Clustering by Passing Messages Between Data Points."
Science Vol. 315, Issue 5814 (2007): 972-976

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. "Hierarchical clustering." The elements of statistical learning 2 (2009).

Jordan, Michael I., Andrew Y. Ng and Yair Weiss: "On spectral clustering: Analysis and an algorithm." In: Advances in Neural Information Processing Systems. 2, (2002), S. 849-856.

Lloyd, S. P. "Least square quantization in PCM". Bell Telephone Laboratories Paper. Published later: Lloyd., S. P. (1982). "Least squares quantization in PCM." IEEE Transactions on Information Theory. 28 (1957): 129–137.

MacQueen, J. B. "Some Methods for classification and Analysis of Multivariate Observations." Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability. 1. University of California Press (1967): 281–297.

Malik, Jitendra and Jianbo Shi. "Normalized Cuts and Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence." 22(8), (2000), S. 888-905.

Ostrovsky, R., Rabani, Y., Schulman, L. J. and Swamy, C. "The Effectiveness of Lloyd-Type Methods for the k-Means Problem". Proceedings of the 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS'06). IEEE. (2006) pp. 165–174.

Perez-Hernandez, Guillermo, Fabian Paul, Toni Giorgino, Gianni de Fabritiis and Frank Noé.
"Identification of slow molecular order parameters for Markov model construction" Journal of Chemical Physics, 139(1) (2013): 015102. 

Prinz, J.-H., H. Wu, M. Sarich, B. Keller, M. Senne, M. Held, J. D. Chodera, C. Schütte and F. Noé: "Markov models of molecular kinetics: Generation and Validation." J. Chem. Phys. 134, 174105 (2011).

Strehl, Alexander, and Joydeep Ghosh. "Cluster ensembles---a knowledge reuse framework for combining multiple partitions." Journal of machine learning research 3.Dec (2002): 583-617.

Turlach, Berwin A. "Bandwidth selection in kernel density estimation: A review." Louvain-la-Neuve: Université catholique de Louvain, 1993.
