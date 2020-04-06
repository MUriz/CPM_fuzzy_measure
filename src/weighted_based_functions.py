import numpy as np
from sklearn import metrics
from metrics import tpr_mean, gm, get_f_measure_function, get_auc_score_function, get_ap_score_function


class WeightedBasedFunction(object):
    """
    The parent class for all Weighted Based Function (AM, GM, HM and Ordered ones)
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


class WeightedBasicFunction(WeightedBasedFunction):
    """
    Parent class of Weighted Basic Based Function (AM, GM, HM, ...)
    """

    def __init__(self, ensemble_matrix, target, score="acc"):
        """
        Constructor
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: The target (true class) of each example
        :param score: The score function (performance) to use for weight calculation
        Valid scores: 'acc', 'tpr', 'gm', 'f1', 'auc' and 'ap'
        """
        # ensemble_matrix => num_classifiers, num_classes, num_instances
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        # Create an empty array
        performances = np.empty((num_classifiers,))
        # Get the callable function
        if score == "acc":
            performance = metrics.accuracy_score
        elif score == "tpr":
            performance = tpr_mean
        elif score == "gm":
            performance = gm
        elif score == "f1":
            performance = get_f_measure_function(target)
        elif score == "auc":
            performance = get_auc_score_function(target)
        elif score == "ap":
            performance = get_ap_score_function(target)
        else:
            raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
        # For each classifier, get the performance
        for i in range(num_classifiers):
            # For auc and ap, the needed value is the probability of being of positive class
            if score == "auc" or score == "ap":
                val = performance(target, ensemble_matrix[i, :, :].T)
            else:
                # For the other metrics, the predicted class (max confidence class)
                pred = np.argmax(ensemble_matrix[i, :, :], axis=0)
                val = performance(target, pred)
            # Set the performance
            performances[i] = val

        # Store normalized accuracies
        self.weights = performances / np.sum(performances)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        raise NotImplementedError("Called aggregate on parent abstract class")


class WeightedAverageFunction(WeightedBasicFunction):
    """
    Class for the Weighted Average Aggregation
    """

    def __init__(self, ensemble_matrix, target, score="acc"):
        """
        Constructor
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: The target (true class) of each example
        :param score: The score function (performance) to use for weight calculation
        Valid scores: 'acc', 'tpr', 'gm', 'f1', 'auc' and 'ap'
        """
        super().__init__(ensemble_matrix, target, score)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        # Create empty array
        agg = np.empty((num_instances, num_classes))
        # For each example
        for i in range(num_instances):
            # For each class
            for c in range(num_classes):
                # Multiply weights and classifiers confidence element wise
                agg[i, c] = np.sum(ensemble_matrix[:, c, i] * self.weights)
        return agg


class WeightedGMFunction(WeightedBasicFunction):
    """
    Class fot the Weighted GM
    """

    def __init__(self, ensemble_matrix, target, score="acc"):
        """
        Constructor
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: The target (true class) of each example
        :param score: The score function (performance) to use for weight calculation
        Valid scores: 'acc', 'tpr', 'gm', 'f1', 'auc' and 'ap'
        """
        super().__init__(ensemble_matrix, target, score)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                # Power each element by thw weight and multiply them all
                agg[i, c] = np.prod(np.power(ensemble_matrix[:, c, i], self.weights))
        return agg


class WeightedHMFunction(WeightedBasicFunction):
    """
    Class fot the Weighted HM
    """

    def __init__(self, ensemble_matrix, target, score="acc"):
        """
        Constructor
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: The target (true class) of each example
        :param score: The score function (performance) to use for weight calculation
        Valid scores: 'acc', 'tpr', 'gm', 'f1', 'auc' and 'ap'
        """
        super().__init__(ensemble_matrix, target, score)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                if np.any(ensemble_matrix[:, c, i] == 0):
                    agg[i, c] = 0.0
                else:
                    agg[i, c] = 1.0 / np.sum(self.weights / ensemble_matrix[:, c, i])
        return agg


class OWAFunction(WeightedBasedFunction):
    """
    Class for the OWA functions
    """
    def __init__(self, ensemble_matrix, a, b):
        """
        Constructor
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param a: The 'a' parameter for Q function used for weight calculation
        :param b: The 'b' parameter for Q function used for weight calculation
        """
        self.a = a
        self.b = b
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        # Create empty weight array
        self.weights = np.empty((num_classifiers,))
        # Calculate weights using Q function
        for i in range(1, num_classifiers + 1):
            self.weights[i - 1] = self.Q(float(i) / float(num_classifiers)) - \
                                  self.Q(float(i - 1) / float(num_classifiers))

    def Q(self, r):
        """
        Auxiliary function for weight calculation
        :param r: The value used for weight calculation
        :return:
        """
        if r < self.a:
            return 0.0
        elif self.a <= r <= self.b:
            return float(r - self.a) / float(self.b - self.a)
        else:
            assert r > self.b
            return 1.0

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                # Sort values first
                v = np.sort(ensemble_matrix[:, c, i])[::-1]
                # Multiply element wise
                agg[i, c] = np.sum(self.weights * v)
        return agg
