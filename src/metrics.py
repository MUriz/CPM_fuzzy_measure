from sklearn import metrics as skl_metrics
import numpy as np
from numba import jit
from collections import Counter
import math


def get_f_measure_function(true_target):
    """
    Function that returns the f1 score function, but setting properly the positive and negative classes
    Positive class: The less represented class
    Negative class: The most represented class
    :param true_target: An array with the true class (0 or 1) for each example
    :return: A callable function that calculates the f1 score given the target and prediction (this order)
    """
    # Assertions: There are 2 classes and there are 0 and 1
    unique_classes = np.unique(true_target)
    assert unique_classes.shape[0] == 2
    assert np.min(unique_classes) == 0
    assert np.max(unique_classes) == 1
    # Count 0s and 1s
    num_c0 = sum(true_target == 0)
    num_c1 = sum(true_target == 1)
    # If num_c1 <= num_c0: positive class c1, else c0
    if num_c1 <= num_c0:
        return lambda target, pred: f_measure(target, pred, positive_class=1, negative_class=0)
    else:
        return lambda target, pred: f_measure(target, pred, positive_class=0, negative_class=1)


def get_auc_score_function(true_target):
    """
    Function that returns the auc score function, but setting properly the positive and negative classes
    Positive class: The less represented class
    Negative class: The most represented class
    :param true_target: An array with the true class (0 or 1) for each example
    :return: A callable function that calculates the auc score given the target and prediction (this order)
    """
    # Assertions: There are 2 classes and there are 0 and 1
    unique_classes = np.unique(true_target)
    assert unique_classes.shape[0] == 2
    assert np.min(unique_classes) == 0
    assert np.max(unique_classes) == 1
    # Count 0s and 1s
    num_c0 = sum(true_target == 0)
    num_c1 = sum(true_target == 1)
    # If num_c1 <= num_c0: positive class c1, else c0
    if num_c1 <= num_c0:
        positive_class = 1
    else:
        positive_class = 0

    # Define the function
    def score(target, agg):
        return skl_metrics.roc_auc_score((target == positive_class).astype('int'), agg[:, positive_class])

    return score


def get_ap_score_function(true_target):
    """
    Function that returns the ap (average precision) function, but setting properly the positive and negative classes
    Positive class: The less represented class
    Negative class: The most represented class
    :param true_target: An array with the true class (0 or 1) for each example
    :return: A callable function that calculates the ap score given the target and prediction (this order)
    """
    # Assertions: There are 2 classes and there are 0 and 1
    unique_classes = np.unique(true_target)
    assert unique_classes.shape[0] == 2
    assert np.min(unique_classes) == 0
    assert np.max(unique_classes) == 1
    # Count 0s and 1s
    num_c0 = sum(true_target == 0)
    num_c1 = sum(true_target == 1)
    # If num_c1 <= num_c0: positive class c1, else c0
    if num_c1 <= num_c0:
        positive_class = 1
    else:
        positive_class = 0

    # Define the function
    def score(target, agg):
        return skl_metrics.average_precision_score((target == positive_class).astype('int'), agg[:, positive_class])

    return score


@jit
def confusion_matrix(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the confusion matrix
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: TP, TN, FP and FN
    """
    n = y_true.shape[0]
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        if y_true[i] == positive_class:
            if y_true[i] == y_pred[i]:
                tp += 1
            else:
                fn += 1
        elif y_true[i] == negative_class:
            if y_true[i] == y_pred[i]:
                tn += 1
            else:
                fp += 1
        else:
            raise Exception("Unexcepted class")

    return tp, tn, fp, fn


def precision(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the precision
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: Precision value
    """
    # Get CM
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred, positive_class, negative_class)
    # Calculate precision
    return 0.0 if tp == 0 else float(tp) / float(tp + fp)


def f_measure(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the f1 score
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: f1 score value
    """
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred, positive_class, negative_class)
    return (2.*tp) / (2.*tp + fp + fn)


def false_positive_rate(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the FPR
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: FPR
    """
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred, positive_class, negative_class)
    return float(fp) / float(tn + fp)


def tnr_and_tpr(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the TPR and TNR
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: TNR and TPR
    """
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred, positive_class, negative_class)
    tpr = 0.0 if tp == 0 else float(tp) / float(tp + fn)
    tnr = 0.0 if tn == 0 else float(tn) / float(tn + fp)
    return tnr, tpr

@jit
def accuracy(y_true, y_pred):
    """
    Function that calculates the accuracy
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :return: Accuracy
    """
    n = y_true.shape[0]
    n_ok = 0.0
    for i in range(n):
        if y_true[i] == y_pred[i]:
            n_ok += 1
    return n_ok / n

@jit
def tpr_mean(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the mean of the tpr and tnr
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: TPR and TNR mean value
    """
    tprs = tnr_and_tpr(y_true, y_pred, positive_class, negative_class)
    return (tprs[0] + tprs[1])/2

@jit
def gm(y_true, y_pred, positive_class=1, negative_class=0):
    """
    Function that calculates the geometric mean of the tpr and tnr
    :param y_true: The true class array
    :param y_pred: The predicted class array
    :param positive_class: The value of positive class (default: 1)
    :param negative_class: The value of negative class (default: 0)
    :return: GM of tpr and tnr
    """
    tprs = tnr_and_tpr(y_true, y_pred, positive_class, negative_class)
    return math.sqrt(tprs[0]*tprs[1])
