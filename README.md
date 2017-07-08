

## Clustertools

### Overview

A self-contained, scikit-learn API-oriented Python package supporting a variety of different clustering algorithms for Machine Learning applications.

Developed as part of "Mathematical aspects in Machine Learning", Freie Universität Berlin, summer semester 2017.

![](https://raw.githubusercontent.com/clustertoolsgroup/clustertools_project/master/images/meanshift_examples.png)

### Contents
This package provides the following functionality:

Distance-based clustering:

- Regspace
- K-Means(++)

Density-based clustering:

- DBSCAN
- Mean shift

Similarity/graph-based clustering:

- Spectral clustering (Normalized cuts/standardized graph Laplacian)
- Affinity Propagation 

Fuzzy methods:

- Fuzzy C-Means

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


