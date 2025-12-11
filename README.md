# Geometry-Aware Dataset Diversity via Persistence Landscapes

A reference implementation accompanying the paper: Geometry-Aware Metric for Dataset Diversity via Persistence Landscapes.

This repository provides a complete implementation of PLDiv, a geometry-aware diversity metric based on persistent homology and persistence landscapes, along with all experiment code from the paper. The metric captures both geometric and topological structures in data, addressing the limitations of entropy-based diversity metrics and traditional dispersion measures.

PLDiv is theoretically grounded, satisfies key diversity axioms (effective size, monotonicity, twin property, symmetry), and demonstrates strong empirical performance across text, images, and synthetic geometric point clouds.



## Quick Start
Compute PLDiv from a point cloud

``` python
from PLDiv import compute_pldiv
import numpy as np

X = np.random.randn(200, 2)  # point cloud
score = compute_pldiv(X)
print("PLDiv:", score)
```

Compute PLDiv from a distance matrix

``` python
from PLDiv import compute_pldiv
import numpy as np
from sklearn.metrics import pairwise_distances

X = np.random.randn(200, 20)

# Example: Euclidean distance matrix
D = pairwise_distances(X, metric='euclidean')
score = compute_pldiv(D, distance_matrix = True)
print("PLDiv:", score)
```


Compute Sparse PLDiv (recommended for large datasets)

``` python
from PLDiv_sparse import compute_sparse_pldiv

score = compute_sparse_pldiv(distance_matrix, sparse = 0.3)
print("Sparse PLDiv:", score)
```

