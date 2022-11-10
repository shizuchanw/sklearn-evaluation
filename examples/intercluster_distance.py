from sklearn_evaluation import plot

# import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



X, y = make_blobs(n_samples=1000, n_features=4, centers=12, random_state=42)

plot.intercluster_distance(X)