
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
numpy, matplotlib

### Installation
Call
```python setup.py install```
from project directory. Alternatively, call
```
pip install git+https://github.com/clustertoolsgroup/clustertools_project
``` 
to install from github.

### References

Arthur, D. and Vassilvitskii, S. "k-means++: the advantages of careful seeding." 
Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035 (2007).

Bezdek, James C., Robert Ehrlich, and William Full. "FCM: The fuzzy c-means clustering algorithm." Computers \& Geosciences 10.2-3 (1984): 191-203

Comaniciu, Dorin, and Peter Meer. "Mean shift: A robust approach toward feature space analysis." IEEE Transactions on pattern analysis and machine intelligence 24.5 (2002): 603-619.

Frey, Brendan J. and Delbert Dueck. "Clustering by Passing Messages Between Data Points."
Science Vol. 315, Issue 5814 (2007): 972-976

Jordan, Michael I., Andrew Y. Ng and Yair Weiss: "On spectral clustering: Analysis and an algorithm." In: Advances in Neural Information Processing Systems. 2, (2002), S. 849-856.

Malik, Jitendra and Jianbo Shi. "Normalized Cuts and Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence." 22(8), (2000), S. 888-905.

Ostrovsky, R., Rabani, Y., Schulman, L. J. and Swamy, C. "The Effectiveness of Lloyd-Type Methods for the k-Means Problem". 
Proceedings of the 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS'06). IEEE. (2006) pp. 165–174.

Perez-Hernandez, Guillermo, Fabian Paul, Toni Giorgino, Gianni de Fabritiis and Frank Noé.
"Identification of slow molecular order parameters for Markov model construction" Journal of Chemical Physics, 139(1) (2013): 015102. 

Prinz, J.-H., H. Wu, M. Sarich, B. Keller, M. Senne, M. Held, J. D. Chodera, C. Schütte and F. Noé: "Markov models of molecular kinetics: Generation and Validation." J. Chem. Phys. 134, 174105 (2011).

Turlach, Berwin A. "Bandwidth selection in kernel density estimation: A review." Louvain-la-Neuve: Université catholique de Louvain, 1993.


As well as datasets from https://cs.joensuu.fi/sipu/datasets/ (07/07/2017)

