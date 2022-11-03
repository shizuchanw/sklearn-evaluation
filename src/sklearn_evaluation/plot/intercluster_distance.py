"""
Plots for intercluster distance.
"""

from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn.cluster import KMeans


@SKLearnEvaluationLogger.log(feature='plot')
def intercluster_distance(X,
                          model=KMeans(),
                          n_clusters=6,
                          ax=None):
    """Plots elbow curve of different values of K of a clustering algorithm.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]:
        Data to cluster, where n_samples is the number of samples and
        n_features is the number of features.

    model
        Clusterer instance that implements ``fit``,``fit_predict``, and
        ``score`` methods, and an ``n_clusters`` hyperparameter.
        e.g. :class:`sklearn.cluster.KMeans` instance

    n_clusters : None or :obj:`list` of int, optional
        List of n_clusters for which to plot the explained variances.
        Defaults to ``[1, 3, 5, 7, 9, 11]``.

    ax : :class:`matplotlib.axes.Axes`, optional
        The axes upon which to plot the curve. If None, the plot is drawn
        on the current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/elbow_curve.py

    """

    if not hasattr(model, 'n_clusters'):
        raise TypeError('"n_clusters" attribute not in classifier. '
                        'Cannot plot intercluster distance.')

    if ax is None:
        ax = plt.gca()

    model.n_clusters = n_clusters;
    model.fit(X);


    ax.set_title('Intercluster Distance')
    ax.plot(n_clusters, sum_of_squares, 'b*-')
    ax.grid(True)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Sum of Squared Errors')
    return
