import numpy as np
from sklearn.metrics import precision_score
from . import validate


@validate.argument_is_proportion('top_proportion')
def precision_at(y_true, y_score, top_proportion, ignore_nas=False):
    '''
    Calculates precision at a given proportion.
    Only supports binary classification.
    '''
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]

    # Based on the proportion, get the index to split the data
    # if value is negative, return 0
    cutoff_index = max(int(len(y_true) * top_proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]

    # Convert scores to binary, by comparing them with the cutoff value
    scores_binary = np.array([int(y >= cutoff_value) for y in y_score])
    # Calculate precision using sklearn function
    if ignore_nas:
        precision = __precision(y_true, scores_binary)
    else:
        precision = precision_score(y_true, scores_binary)

    return precision, cutoff_value


@validate.argument_is_proportion('top_proportion')
def cutoff_score_at_top_proportion(y_score, top_proportion):
    """
    Sort scores and get the score at
    """
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]
    # Based on the proportion, get the index to split th
    # if value is negative, return 0
    cutoff_index = max(int(len(y_score) * top_proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]
    return cutoff_value


@validate.argument_is_proportion('top_proportion')
def binarize_scores_at_top_proportion(y_score, top_proportion):
    """Binary scores grabbing the top scores
    """
    cutoff_score = cutoff_score_at_top_proportion(y_score, top_proportion)
    y_score_binary = np.array([int(y >= cutoff_score) for y in y_score])
    return y_score_binary


@validate.argument_is_proportion('quantile')
def binarize_scores_at_quantile(y_score, quantile):
    """Binary scores at certain quantile
    """
    cutoff_score = np.quantile(y_score, quantile)
    y_score_binary = (y_score > cutoff_score).astype(int)
    return y_score_binary


def __precision(y_true, y_pred):
    '''
        Precision metric tolerant to unlabeled data in y_true,
        NA values are ignored for the precision calculation
    '''
    # make copies of the arrays to avoid modifying the original ones
    y_true = np.copy(y_true)
    y_pred = np.copy(y_pred)

    # precision = tp/(tp+fp)
    # True nehatives do not affect precision value, so for every missing
    # value in y_true, replace it with 0 and also replace the value
    # in y_pred with 0
    is_nan = np.isnan(y_true)
    y_true[is_nan] = 0
    y_pred[is_nan] = 0
    precision = precision_score(y_true, y_pred)
    return precision


@validate.argument_is_proportion('top_proportion')
def tp_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    tp = (y_pred == 1) & (y_true == 1)
    return tp.sum()


@validate.argument_is_proportion('top_proportion')
def fp_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    fp = (y_pred == 1) & (y_true == 0)
    return fp.sum()


@validate.argument_is_proportion('top_proportion')
def tn_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    tn = (y_pred == 0) & (y_true == 0)
    return tn.sum()


@validate.argument_is_proportion('top_proportion')
def fn_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    fn = (y_pred == 0) & (y_true == 1)
    return fn.sum()


@validate.argument_is_proportion('top_proportion')
def labels_at(y_true, y_score, top_proportion, normalize=False):
    '''
        Return the number of labels encountered in the top  X proportion
    '''
    # Get indexes of scores sorted in descending order
    indexes = np.argsort(y_score)[::-1]

    # Sort true values in the same order
    y_true_sorted = y_true[indexes]

    # Grab top x proportion of true values
    cutoff_index = max(int(len(y_true_sorted) * top_proportion) - 1, 0)
    # add one to index to grab values including that index
    y_true_top = y_true_sorted[:cutoff_index+1]

    # Count the number of non-nas in the top x proportion
    # we are returning a count so it should be an int
    values = int((~np.isnan(y_true_top)).sum())

    if normalize:
        values = float(values)/(~np.isnan(y_true)).sum()

    return values
