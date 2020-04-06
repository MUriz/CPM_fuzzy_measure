import numpy as np
import scipy.stats


class UnweightedFunction(object):
    """
    Parent class for Unweighted functions
    """

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        raise NotImplementedError("Called aggregate on parent abstract class")


class MeanFunction(UnweightedFunction):
    """
    Class for Arithmetic Mean function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        ag = np.empty((num_instances, num_classes))
        # For each class and instance, get the mean of the confidences
        for c in range(num_classes):
            ag[:, c] = np.mean(ensemble_matrix[:, c, :], axis=0)
        return ag


class MedianFunction(UnweightedFunction):
    """
    Class for Median function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        ag = np.empty((num_instances, num_classes))
        # For each class and instance, get the median of the confidences
        for c in range(num_classes):
            ag[:, c] = np.median(ensemble_matrix[:, c, :], axis=0)
        return ag


class GeometricMeanFunction(UnweightedFunction):
    """
    Class for Geometric Mean function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        np.seterr(invalid='raise')
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg = np.empty((num_instances, num_classes))
        # Multiply all confidences and power the result by 1/num_classifiers
        for i in range(num_instances):
            for c in range(num_classes):
                v = ensemble_matrix[:, c, i]
                try:
                    agg[i, c] = np.power(np.prod(v), 1./num_classifiers)
                except FloatingPointError:
                    print(v)
                    return None
        return agg


class HarmonicMeanFunction(UnweightedFunction):
    """
    Class for HM function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg_vector = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                v = ensemble_matrix[:, c, i]
                try:
                    agg_vector[i, c] = scipy.stats.hmean(v)
                except ValueError:
                    agg_vector[i, c] = 0.0
        return agg_vector


class MaxFunction(UnweightedFunction):
    """
    Class for Max function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        ag = np.empty((num_instances, num_classes))
        for c in range(num_classes):
            ag[:, c] = np.max(ensemble_matrix[:, c, :], axis=0)
        return ag


class MinFunction(UnweightedFunction):
    """
    Class for Min function
    """
    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        ag = np.empty((num_instances, num_classes))
        for c in range(num_classes):
            ag[:, c] = np.min(ensemble_matrix[:, c, :], axis=0)
        return ag
