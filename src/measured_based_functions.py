import itertools

import numpy as np

import DataStore
import choquet
import sugeno
from metrics import tpr_mean, gm, accuracy, get_f_measure_function, get_auc_score_function, get_ap_score_function


def all_combs(values):
    """
    Function (iterator) that given an array of elements calculates all it's combinations
    f.e.: values=[1,2,3] it will return the elements [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
    NOTE: All values are tuples, including the one elements values
    :param values:
    :return: Iterator, it yields each combination
    """
    n = len(values)
    # Yield the given values but in tuple format
    for i in values:
        yield (i,)
    # For 1 to N
    for i in range(1, n):
        # Get the combinations using (i+1) elements
        for j in itertools.combinations(values, i + 1):
            # Yield it
            yield j


def m_function_squared_mean(v):
    """
    The m function used in OIFM method
    :param v: array of values
    :return: The value after apply the squared mean function
    """
    return np.sqrt(np.mean(np.power(v, 2)))


def build_measure(ensemble_matrix, target, score="acc"):
    """
    Function that builds the CPM measure
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param score: The score to use for classifier performance calculation
    :return: The additive measure
    """
    # ensemble_matrix => num_classifiers, num_instances
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    # Store the performances
    performances = dict()
    # Store the mean of each level
    level_mean = dict()
    # Get callable performance function
    if score == "acc":
        # performance_function = metrics.accuracy_score
        performance_function = accuracy
    elif score == "tpr":
        performance_function = tpr_mean
    elif score == "gm":
        performance_function = gm
    elif score == "f1":
        performance_function = get_f_measure_function(target)
    elif score == "auc":
        performance_function = get_auc_score_function(target)
    elif score == "ap":
        performance_function = get_ap_score_function(target)
    else:
        raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
    # for each possible classifier combination
    for i in all_combs(range(num_classifiers)):
        classifiers_prob = ensemble_matrix[i, :, :]
        # Mean of probabilities
        prob = np.mean(classifiers_prob, axis=0)
        if score == "auc" or score == "ap":
            val = performance_function(target, prob.T)
        else:
            # prob => num_classes, num_instances
            pred = np.argmax(prob, axis=0)
            val = performance_function(target, pred)
        # Add performances and store for level mean calculation
        performances[i] = val
        if len(i) not in level_mean:
            level_mean[len(i)] = [0.0, 0.0]
        level_mean[len(i)][0] += val
        level_mean[len(i)][1] += 1.0

    # Calculate the mean per level
    for k in level_mean.keys():
        level_mean[k] = level_mean[k][0] / level_mean[k][1]

    # Calculate the measure
    measure = DataStore.DictDataStore(num_classifiers)
    # For each accuracy set measure value as variation of mean based on difference with the level mean
    for i in all_combs(range(num_classifiers)):
        y = performances[i] - level_mean[len(i)]
        # value = (float(len(i)) / float(num_classifiers)) * (1 + y)
        value = (float(len(i)) / float(num_classifiers)) + np.tanh(y * 100) / (2.0 * num_classifiers)
        measure.put(i, value)

    return measure


def build_measure_additive(ensemble_matrix, target, score="acc"):
    """
    Function that builds the additive measure
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param score: The score to use for classifier performance calculation
    :return: The additive measure
    """
    # ensemble_matrix => num_classifiers, num_instances
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    performances = np.empty(num_classifiers)
    if score == "acc":
        # performance_function = metrics.accuracy_score
        performance_function = accuracy
    elif score == "tpr":
        performance_function = tpr_mean
    elif score == "gm":
        performance_function = gm
    elif score == "f1":
        performance_function = get_f_measure_function(target)
    elif score == "auc":
        performance_function = get_auc_score_function(target)
    elif score == "ap":
        performance_function = get_ap_score_function(target)
    else:
        raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
    # for each possible classifier combination
    for i in range(num_classifiers):
        # Get accuracy of classifiers i
        prob = ensemble_matrix[i, :, :]
        if score == "auc" or score == "ap":
            val = performance_function(target, prob.T)
        else:
            # prob => num_classes, num_instances
            pred = np.argmax(prob, axis=0)
            val = performance_function(target, pred)
        performances[i] = val

    level_mean = performances.mean()
    y = performances - level_mean
    values = (1.0 / num_classifiers) + np.tanh(y * 100) / (2.0 * num_classifiers)

    measure = DataStore.DictDataStore(num_classifiers)
    # For each accuracy set measure value as variation of mean based on difference with the level mean
    for i in all_combs(range(num_classifiers)):
        value = 0.0
        for j in i:
            value += values[j]
        measure.put(i, value)

    measure.normalize()
    return measure


def build_measure_m_squared_mean_aggregation(ensemble_matrix, target, score="acc"):
    """
    Decorator that calls build the measure for the OIFM method using "m_function_squared_mean" function
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param score: The score to use for classifier performance calculation
    :return: The function that builds the measure for the OIFM method with the "m_function_squared_mean" function
    """
    return build_measure_m_aggregation(ensemble_matrix, target, m_function_squared_mean, score)


def build_measure_m_aggregation(ensemble_matrix, target, m_function, score="acc"):
    """
    Returns the measure for the OIFM method
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param m_function: The function to use for measure calculation
    :param score: The score to use for classifier performance calculation
    :return: Measure
    """
    # ensemble_matrix => num_classifiers, num_instances
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    performances = np.empty((num_classifiers,))
    if score == "acc":
        performance_function = accuracy
    elif score == "tpr":
        performance_function = tpr_mean
    elif score == "gm":
        performance_function = gm
    elif score == "f1":
        performance_function = get_f_measure_function(target)
    elif score == "auc":
        performance_function = get_auc_score_function(target)
    elif score == "ap":
        performance_function = get_ap_score_function(target)
    else:
        raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
    # For each individual classifier get its performance
    for i in range(num_classifiers):
        # Get accuracy of classifiers i
        prob = ensemble_matrix[i, :, :]
        if score == "auc" or score == "ap":
            val = performance_function(target, prob.T)
        else:
            pred = np.argmax(prob, axis=0)
            val = performance_function(target, pred)
        performances[i] = val

    measure = DataStore.DictDataStore(num_classifiers)
    # Calculate denominator
    performances_2 = np.power(performances, 2)
    denominator = m_function(performances_2)
    # For each combination get the measure value
    for i in all_combs(range(num_classifiers)):
        v = np.zeros((num_classifiers,))
        for j in i:
            v[j] = performances_2[j]
        nominator = m_function(v)
        measure.put(i, nominator / denominator)

    return measure


def build_measure_GISFM(ensemble_matrix, target, score):
    """
    Function to create the global ISFM measure
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param score: The score to use for classifier performance calculation
    :return:
    """
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    similarities = compute_similarities(ensemble_matrix)
    # To store measure
    measure = DataStore.DictDataStore(num_classifiers)
    confidences = np.empty((num_classifiers,))
    # Get the callable score function
    if score == "acc":
        performance_function = accuracy
    elif score == "tpr":
        performance_function = tpr_mean
    elif score == "gm":
        performance_function = gm
    elif score == "f1":
        performance_function = get_f_measure_function(target)
    elif score == "auc":
        performance_function = get_auc_score_function(target)
    elif score == "ap":
        performance_function = get_ap_score_function(target)
    else:
        raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
    # For each individual classifier get its performance
    for i in range(num_classifiers):
        # Get accuracy of classifiers i
        prob = ensemble_matrix[i, :, :]
        if score == "auc" or score == "ap":
            val = performance_function(target, prob.T)
        else:
            pred = np.argmax(prob, axis=0)
            val = performance_function(target, pred)
        confidences[i] = val
        measure.put((i,), 0.0)
    # Get the order of confidences
    order = np.argsort(confidences)
    # Calculate values
    for i in range(len(order)):
        s = similarities[order[i], order[i + 1:]]
        if len(s) == 0:
            s = 0.0
        else:
            s = s.max()
        measure.put((order[i],), confidences[order[i]] * (1 - s))

    for i in all_combs(range(num_classifiers)):
        if len(i) > 1:
            v = 0.0
            for j in i:
                v += measure.get((j,))
            measure.put(i, v)

    measure.normalize()
    return measure


def compute_similarities(ensemble_matrix):
    """
    Function that computes the similarities
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :return: Similarities
    """
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    prediction = np.argmax(ensemble_matrix, axis=1)
    similarities = np.empty((num_classifiers, num_classifiers))
    for i in range(num_classifiers):
        for j in range(i, num_classifiers):
            similarities[i, j] = np.count_nonzero(prediction[i, :] == prediction[j, :]) / num_instances
            similarities[j, i] = similarities[i, j]
    return similarities


def dynamic_iisfm(values, confidences, similarities):
    """
    Function that calculates the dynamic measure values for the isfm method
    :param values: The values to aggregate
    :param confidences: The confidence of each classifier
    :param similarities: Matrix of num_classifiers by num_classifiers with the similarity of each pair of classifiers
    :return: The measure values
    """
    # Sort by values descending
    order = np.argsort(values)[::-1]
    # Measure values
    measure_values = np.empty((confidences.shape[0],))
    # The first element is the confidence
    measure_values[order[0]] = confidences[order[0]]
    # For others
    for i in range(1, confidences.shape[0]):
        s = similarities[order[i], order[:i]]
        if len(s) == 0:
            s = 0.0
        else:
            s = s.max()
        measure_values[order[i]] = measure_values[order[i - 1]] + confidences[order[i]] * (1 - s)

    return measure_values / measure_values.max() if measure_values.max() != 0 else measure_values


def dynamic_mhm(values, additive_measure, relative_diversity, alpha=1.0):
    """
    Function that calculates the dynamic measure values for the mhm method
    :param values: Values to be aggregated
    :param additive_measure: The additive measure
    :param relative_diversity: The relative diversity
    :param alpha: The alpha value (paper)
    :return: The values of the measure
    """

    # Sort by values descending
    order = np.argsort(values)[::-1]
    # Measure values
    measure_values = np.empty((values.shape[0],))
    # The first element
    measure_values[order[0]] = additive_measure.get((order[0],)) * (1 + alpha * relative_diversity[(order[0],)])
    # For others
    for i in range(1, values.shape[0]):
        subset = tuple(sorted(order[:i + 1]))
        val = additive_measure.get(subset) * (1 + alpha * relative_diversity[subset])
        val = max(measure_values[order[i - 1]], val)
        measure_values[order[i]] = val

    return measure_values / measure_values.max() if measure_values.max() != 0 else measure_values


def build_global_mhm(ensemble_matrix, target, score, alpha=1.0):
    """
    Function to create the global mhm measure
    :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
    :param target: An array with the real class
    :param score: The score to use for classifier performance calculation
    :param alpha: Alpha parameter (paper)
    :return:
    """
    num_classifiers, num_classes, num_instances = ensemble_matrix.shape
    confidences = np.empty((num_classifiers,))
    # For additive measure
    additive_measure = DataStore.DictDataStore(num_classifiers)
    # Get callable score function
    if score == "acc":
        performance_function = accuracy
    elif score == "tpr":
        performance_function = tpr_mean
    elif score == "gm":
        performance_function = gm
    elif score == "f1":
        performance_function = get_f_measure_function(target)
    elif score == "auc":
        performance_function = get_auc_score_function(target)
    elif score == "ap":
        performance_function = get_ap_score_function(target)
    else:
        raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
    # For each individual classifier get its performance
    for i in range(num_classifiers):
        prob = ensemble_matrix[i, :, :]
        if score == "auc" or score == "ap":
            val = performance_function(target, prob.T)
        else:
            pred = np.argmax(prob, axis=0)
            val = performance_function(target, pred)
        confidences[i] = val

    # Calculate additive measure
    for i in all_combs(range(num_classifiers)):
        if len(i) == 1:
            additive_measure.put(i, confidences[i[0]])
        else:
            v = 0.0
            for j in i:
                v += additive_measure.get((j,))
            additive_measure.put(i, v)
    additive_measure.normalize()

    # Compute similarities and relative diversity
    similarities = compute_similarities(ensemble_matrix)
    relative_diversity = relative_diversity_dict(similarities)

    # Calculate the final measure
    measure = DataStore.DictDataStore(num_classifiers)
    for i in all_combs(range(num_classifiers)):
        value = additive_measure.get(i) * (1 + alpha * relative_diversity[i])
        measure.put(i, value)

    measure.correct_monotonicity()
    measure.normalize()
    return measure


def relative_diversity_dict(similarity_matrix):
    """
    Calculate the relative diversity and return it in dictionary
    :param similarity_matrix: num_classifiers by num_classifiers matrix with the similarity of each pair of classifiers
    :return: A dict with the relative diversity of each possible subset of classifiers
    """
    num_classifiers = similarity_matrix.shape[0]
    values = []
    for i in range(num_classifiers):
        for j in range(i + 1, num_classifiers):
            values.append(similarity_matrix[i, j])
    max_disimilarity = 1 - np.min(values)

    relative_diversity = dict()
    for i in all_combs(range(num_classifiers)):
        if len(i) == 1 or max_disimilarity == 0:
            relative_diversity[i] = 0
        else:
            s = 0.0
            for j in range(len(i)):
                for k in range(j + 1, len(i)):
                    s += 1 - similarity_matrix[i[j], i[k]]
            diversity = (2.0 / (np.square(len(i)) - len(i))) * s
            relative_diversity[i] = 2 * diversity / max_disimilarity - 1
    return relative_diversity


def pre_agg_max(x):
    """
    Max pre-aggregation function
    :param x: tuple of two elements (value to aggregate, measure value)
    :return:
    """
    return np.max([0.0, x[0] + x[1] - 1])


class MeasuredBasedFunction(object):
    """
    Class for Measure Based functions
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


class ChoquetBasedFunction(MeasuredBasedFunction):
    """
    Class for Choquet based
    """

    def __init__(self, ensemble_matrix, target, pre_agg_function, build_function=build_measure, score="acc"):
        """

        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: An array with the real class
        :param pre_agg_function: Pre-aggregation function that is applied (original is product)
        :param build_function: The function that is used to build the measure
        :param score: The score to use for classifier performance calculation
        """
        # Build the mesure
        self.measure = build_function(ensemble_matrix, target, score)
        self.pre_agg_function = pre_agg_function

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        # ensemble_matrix => num_classifiers, num_instances
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg_vector = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                v = ensemble_matrix[:, c, i]
                # Get choquet elements
                choquet_parts = choquet.choquet_elements(v, self.measure)
                # Calculate choquet value appliying the pre aggregation function
                agg_vector[i, c] = np.sum(list(map(self.pre_agg_function, choquet_parts)))
        return agg_vector


class SugenoBasedFunction(MeasuredBasedFunction):
    def __init__(self, ensemble_matrix, target, pre_agg_function, build_function=build_measure, score="acc"):
        """

        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: An array with the real class
        :param pre_agg_function: Pre-aggregation function that is applied (original is min)
        :param build_function: The function that is used to build the measure
        :param score: The score to use for classifier performance calculation
        """
        # Build the measure
        self.measure = build_function(ensemble_matrix, target, score)
        self.pre_agg_function = pre_agg_function

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        # ensemble_matrix => num_classifiers, num_instances
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg_vector = np.empty((num_instances, num_classes))
        for i in range(num_instances):
            for c in range(num_classes):
                v = ensemble_matrix[:, c, i]
                # Get sugeno elements
                sugeno_parts = sugeno.sugeno_elements(v, self.measure)
                # Calculate the aggregated value using pre aggregation function
                agg_vector[i, c] = np.max(list(map(self.pre_agg_function, sugeno_parts)))
        return agg_vector


class DynamicMeasureBasedFunction(MeasuredBasedFunction):
    """
    Class for Dynamic Measure Based functions
    """

    def __init__(self, ensemble_matrix, target, dynamic_measure_function, score, integral):
        """

        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: An array with the real class
        :param dynamic_measure_function: The function that returns the mesure values
        :param score: The score to use for classifier performance calculation
        :param integral: The integral to use (choquet or sugeno)
        """
        # Assertions
        assert integral.lower() in ["choquet", "sugeno"]
        assert score.lower() in ["acc", "tpr", "gm", "f1", "auc", "ap"]
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        self.ensemble_matrix = ensemble_matrix
        self.target = target
        self.dynamic_measure_function = dynamic_measure_function
        self.integral = integral.lower()
        # Get callable score
        if score.lower() == "acc":
            performance_function = accuracy
        elif score.lower() == "tpr":
            performance_function = tpr_mean
        elif score == "gm":
            performance_function = gm
        elif score == "f1":
            performance_function = get_f_measure_function(target)
        elif score == "auc":
            performance_function = get_auc_score_function(target)
        elif score == "ap":
            performance_function = get_ap_score_function(target)
        else:
            raise Exception("score must be 'acc', 'tpr', 'gm', 'f1', 'auc' or 'ap'")
        # Calculate the confidence of each classifier
        self.confidences = np.empty((num_classifiers,))
        for i in range(num_classifiers):
            prob = ensemble_matrix[i, :, :]
            if score == "auc" or score == "ap":
                self.confidences[i] = performance_function(target, prob.T)
            else:
                pred = np.argmax(prob, axis=0)
                self.confidences[i] = performance_function(target, pred)
        # Calculate the similarities
        self.similarities = compute_similarities(ensemble_matrix)
        # If the dynamic function is mhm
        if self.dynamic_measure_function == dynamic_mhm:
            # Calculate the relative diversity
            self.relative_diversity = relative_diversity_dict(self.similarities)
            # Calculate the additive measure
            self.additive_measure = DataStore.DictDataStore(self.confidences.shape[0])
            for i in all_combs(range(self.confidences.shape[0])):
                if len(i) == 1:
                    self.additive_measure.put(i, self.confidences[i[0]])
                else:
                    v = 0.0
                    for j in i:
                        v += self.additive_measure.get((j,))
                    self.additive_measure.put(i, v)
            self.additive_measure.normalize()

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
        # For each example
        for i in range(num_instances):
            for c in range(num_classes):
                v = ensemble_matrix[:, c, i]
                indices = np.argsort(v)
                # Calculate the measure values
                if self.dynamic_measure_function == dynamic_mhm:
                    measure_values = self.dynamic_measure_function(v, self.additive_measure, self.relative_diversity)
                else:
                    measure_values = self.dynamic_measure_function(v, self.confidences, self.similarities)
                # Calculate the choquet or sugeno integral
                if self.integral == 'choquet':
                    agg_value = v[indices[0]] * measure_values[indices[0]]
                    for j in range(1, len(indices)):
                        agg_value += (v[indices[j]] - v[indices[j - 1]]) * measure_values[indices[j]]
                else:
                    # sugeno
                    agg_value = np.minimum(v[indices], measure_values[indices]).max()
                agg_vector[i, c] = agg_value
        return agg_vector


class EntropyMeasureBasedFunction(MeasuredBasedFunction):
    """
    Class for the Entropy Measure Based function
    """

    def __init__(self, ensemble_matrix, target, integral):
        """

        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: An array with the real class
        :param integral: The integral to use (choquet or sugeno)
        """
        # Verify that the integral is valid
        assert integral.lower() in ["choquet", "sugeno"]
        # Save params
        self.ensemble_matrix = ensemble_matrix
        self.target = target
        self.integral = integral.lower()
        self.n_classes = len(np.unique(target))
        # Calculate the max entropy
        self.max_entropy = -np.log2(1.0 / self.n_classes)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        assert self.n_classes == num_classes
        agg_vector = np.empty((num_instances, num_classes))
        # For each example
        for i in range(num_instances):
            # To store each entropy
            entropies = np.empty((num_classifiers,))
            # For each classifier
            for j in range(num_classifiers):
                # Calculate the entropy
                prob_values = ensemble_matrix[j, :, i]
                prob_values[prob_values == 0] = 1e-10
                entropy = -(prob_values * np.log2(prob_values)).sum()
                entropies[j] = self.max_entropy - entropy
            # Get the mean entropy and use the formulas given in the paper
            avg_entropy = entropies.mean()
            avg_above = entropies > (1.5 * avg_entropy)
            adj_avg_entropy = np.mean(entropies[~avg_above])
            entropies[avg_above] = adj_avg_entropy

            # Create the measure values normalizing them
            measure_values = entropies / entropies.sum() if entropies.sum() != 0 else entropies
            # For each class
            for c in range(num_classes):
                # Get the classifier confidences
                v = ensemble_matrix[:, c, i]
                # Get the order
                indices = np.argsort(v)
                # Calculate the choquet integral
                if self.integral == "choquet":
                    agg_value = v[indices[0]] * measure_values[indices].sum()
                    for j in range(1, len(indices)):
                        agg_value += (v[indices[j]] - v[indices[j - 1]]) * measure_values[indices[j:]].sum()
                else:
                    # Calculate the sugeno integral
                    agg_value = np.minimum(v[indices[0]], measure_values[indices].sum())
                    for j in range(1, len(indices)):
                        agg_value = np.maximum(agg_value, np.minimum(v[indices[j]], measure_values[indices[j:]].sum()))

                # Store the aggregated value
                agg_vector[i, c] = agg_value

        return agg_vector


class LambdaFuzzyIntegralFunction(MeasuredBasedFunction):
    """
    Class for Lambda Fuzzy integral
    """

    def calculate_lambda(self, g):
        """
        Function that calculates the lambda value (code from paper)
        :param g: Vector of idividual classifier strength
        :return: The lambda value
        """
        gb = np.prod(g) * np.poly(-1.0 / g)
        gb[-2:] = gb[-2:] - 1
        gc = np.roots(gb)
        gcc = gc.astype(np.complex)
        # The unique real root greater than -1
        ii = (np.imag(gcc) == 0) & (gc >= -1.0) & (np.abs(gc) > 1e-6)
        lmb = np.real(gc[ii])
        # If there is no lambda, return 0.0
        if lmb.shape[0] != 1:
            return 0.0
        else:
            return lmb[0]

    def choquet(self, xs, g):
        """
        Function that calculates the choquet integral value
        :param xs: Sorted values to aggregate
        :param g: The measure values
        :return: The choquet integral value
        """
        L = xs.shape[0]
        agg_value = xs[L-1]
        # From (L-2) to 0
        for k in range(L-2, -1, -1):
            agg_value += (xs[k] - xs[k+1]) * g[k]
        return agg_value

    def sugeno(self, xs, g):
        """
        Function that calculates the sugeno integral value
        :param xs: Sorted values to aggregate
        :param g: The measure values
        :return: The sugeno integral value
        """
        agg_value = min(xs[0], g[0])
        for t in range(1, xs.shape[0]):
            v = min(xs[t], g[t])
            agg_value = max(agg_value, v)
        return agg_value

    def sub_aggregate(self, x):
        """
        Function that aggregates the x array
        :param x: Array to aggregate
        :return: The aggregated value
        """
        # Sort descent
        k = np.argsort(x)[::-1]
        # Sort g too
        gs = self.g[k]
        # new g
        new_g = np.empty(x.shape)
        new_g[0] = gs[0]
        for t in range(1, x.shape[0]):
            new_g[t] = gs[t] + new_g[t - 1] + self.lmb * gs[t] * new_g[t - 1]
        # Calculate aggregated value with the stored integral function
        agg_value = self.integral_function(x[k], new_g)
        return agg_value

    def __init__(self, ensemble_matrix, target, score='acc', integral="choquet"):
        """

        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        :param target: An array with the real class
        :param score: The performance function to use
        :param integral: The integral to use (choquet or sugeno)
        """
        # ensemble_matrix: (num_classifiers, num_classes, num_instances)
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        # Calculate the strength for each classifier
        self.g = np.empty((num_classifiers, ))
        self.integral = integral
        if self.integral == "choquet":
            self.integral_function = self.choquet
        elif self.integral == "sugeno":
            self.integral_function = self.sugeno
        else:
            raise Exception("integral must be choquet or sugeno")
        if score == "acc":
            performance = accuracy
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
            raise Exception("'score' must be acc, tpr, gm, f1, auc or ap")
        # Calculate performances
        for i in range(num_classifiers):
            if score == "auc" or score == "ap":
                self.g[i] = performance(target, ensemble_matrix[i, :, :].T)
            else:
                pred = np.argmax(ensemble_matrix[i, :, :], axis=0)
                self.g[i] = performance(target, pred)

        # Calculate the mean, and apply CPM
        gmean = np.mean(self.g)
        gdiff = self.g - gmean
        self.g = np.ones((num_classifiers,)) / float(num_classifiers)
        self.g = self.g + np.tanh(gdiff * 100) / (2.0 * num_classifiers)

        # Calculate lambda
        self.lmb = self.calculate_lambda(self.g)

    def aggregate(self, ensemble_matrix):
        """
        The method that aggregates the confidence matrix
        :param ensemble_matrix: A numpy array of num_classifiers by num_classes by num_instances
        where each element is the confidence that the classifier gives to the instance to the class
        :return: An array of num_instances by num_classes where each element is the aggregated confidence
        for the instance and class
        """
        num_classifiers, num_classes, num_instances = ensemble_matrix.shape
        agg_values = np.empty((num_instances, num_classes))
        # For each instance
        for i in range(num_instances):
            # Calculate the aggregated value for each class
            agg = np.empty((num_classes, ))
            for c in range(num_classes):
                agg[c] = self.sub_aggregate(ensemble_matrix[:, c, i])
            agg_values[i, :] = agg
        return agg_values
