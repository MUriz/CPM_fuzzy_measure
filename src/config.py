import unweighted_functions
import weighted_based_functions
import measured_based_functions
import numpy as np
import os
import io_utils


def check_config(config):
    """
    Function that ensures that the required parameters are present
    :param config: Dictionary with the configuration
    :return: None. Raises exception if some required parameter is not present
    """
    required_parameters = ["train_data", "train_conf", "train_out", "test_data", "test_conf", "test_out", "method",
                           "num_classifiers", "train_pred", "test_pred"]
    for i in required_parameters:
        if i not in config:
            err = "'%s' parameter not found in config. Required parameters: [%s]" % (i, ",".join(required_parameters))
            raise Exception(err)

    # Check if the given files exist
    if not os.path.isfile(config["train_data"]):
        raise Exception("'train_data' file: '%s' not found" % config["train_data"])
    if not os.path.isfile(config["train_conf"]):
        raise Exception("'train_conf' file: '%s' not found" % config["train_conf"])
    if not os.path.isfile(config["train_out"]):
        raise Exception("'train_out' file: '%s' not found" % config["train_out"])
    if not os.path.isfile(config["test_data"]):
        raise Exception("'test_data' file: '%s' not found" % config["test_data"])
    if not os.path.isfile(config["test_conf"]):
        raise Exception("'test_conf' file: '%s' not found" % config["test_conf"])
    if not os.path.isfile(config["test_out"]):
        raise Exception("'test_out' file: '%s' not found" % config["test_out"])

    # Check that the num_classifiers is an integer > 0
    try:
        nc = int(config["num_classifiers"])
        if nc <= 0:
            raise Exception("The number of classifiers must be > 0")
    except ValueError:
        raise Exception("Can't convert num_classifiers ('%s') to integer" % config["num_classifiers"])


def get_aggregator(config):
    """
    Function that returns the aggregator given the configuration
    :param config: Fictionary with the configuration
    :return: An object with the method "aggregate"
    """
    # Check config
    check_config(config)
    # Read data
    matrix, target = io_utils.read_matrix_and_target(config["train_data"], config["train_conf"], config["train_out"])
    # Get the number of classifiers
    num_classifiers = int(config["num_classifiers"])

    # Check that the number of classifiers is lower than the total num of classifiers
    if num_classifiers > matrix.shape[0]:
        raise Exception("There are only %d classifiers in data and %d are given" % (matrix.shape[0], num_classifiers))

    # Get only data of the first num_classifiers
    matrix = matrix[:num_classifiers, :, :]

    # Create and return the aggregator
    return build_aggregator(matrix, target, config)


def get_choquet_params(config):
    """

    :param config:
    :return:
    """
    params = {}
    if "score" in config:
        params["score"] = config["score"]
    if "pre_agg_function" in config:
        if config["pre_agg_function"] == "prod":
            params["pre_agg_function"] = np.prod
        elif config["pre_agg_function"] == "min":
            params["pre_agg_function"] = np.min
        elif config["pre_agg_function"] == "max":
            params["pre_agg_function"] = measured_based_functions.pre_agg_max,
        else:
            raise Exception("'%s' pre_agg_function not recognized" % config["pre_agg_function"])
    else:
        # Default product
        params["pre_agg_function"] = np.prod
    return params


def get_sugeno_params(config):
    """

    :param config:
    :return:
    """
    params = {}
    if "score" in config:
        params["score"] = config["score"]
    if "pre_agg_function" in config:
        if config["pre_agg_function"] == "min":
            params["pre_agg_function"] = np.min
        elif config["pre_agg_function"] == "prod":
            params["pre_agg_function"] = np.prod
        elif config["pre_agg_function"] == "max":
            params["pre_agg_function"] = measured_based_functions.pre_agg_max,
        else:
            raise Exception("'%s' pre_agg_function not recognized" % config["pre_agg_function"])
    else:
        # Default min
        params["pre_agg_function"] = np.min
    return params


def build_aggregator(ensemble_matrix, target, config):
    """

    :param ensemble_matrix:
    :param target:
    :param config:
    :return:
    """
    if config["method"] == "AM":
        return unweighted_functions.MeanFunction()
    elif config["method"] == "MED":
        return unweighted_functions.MedianFunction()
    elif config["method"] == "GM":
        return unweighted_functions.GeometricMeanFunction()
    elif config["method"] == "HM":
        return unweighted_functions.HarmonicMeanFunction()
    elif config["method"] == "MIN":
        return unweighted_functions.MinFunction()
    elif config["method"] == "MAX":
        return unweighted_functions.MaxFunction()
    elif config["method"] == "WAM":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return weighted_based_functions.WeightedAverageFunction(ensemble_matrix, target, **params)
    elif config["method"] == "WGM":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return weighted_based_functions.WeightedGMFunction(ensemble_matrix, target, **params)
    elif config["method"] == "WHM":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return weighted_based_functions.WeightedHMFunction(ensemble_matrix, target, **params)
    elif config["method"] == "Qalh":
        return weighted_based_functions.OWAFunction(ensemble_matrix, a=0.0, b=0.5)
    elif config["method"] == "Qamap":
        return weighted_based_functions.OWAFunction(ensemble_matrix, a=0.5, b=1.0)
    elif config["method"] == "Qmot":
        return weighted_based_functions.OWAFunction(ensemble_matrix, a=0.3, b=0.8)
    elif config["method"] == "Cm":
        # Get params
        params = get_choquet_params(config)
        return measured_based_functions.ChoquetBasedFunction(ensemble_matrix, target,
                                                             build_function=measured_based_functions.build_measure,
                                                             **params)
    elif config["method"] == "Cm+":
        # Get params
        params = get_choquet_params(config)
        return measured_based_functions.ChoquetBasedFunction(ensemble_matrix, target,
                                                             build_function=measured_based_functions.build_measure_additive,
                                                             **params)
    elif config["method"] == "Cml":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.LambdaFuzzyIntegralFunction(ensemble_matrix, target, integral="choquet",
                                                                    **params)
    elif config["method"] == "Coifm":
        # Get params
        params = get_choquet_params(config)
        return measured_based_functions.ChoquetBasedFunction(ensemble_matrix, target,
                                                             build_function=measured_based_functions.build_measure_m_squared_mean_aggregation,
                                                             **params)
    elif config["method"] == "Cg-isfm":
        # Get params
        params = get_choquet_params(config)
        return measured_based_functions.ChoquetBasedFunction(ensemble_matrix, target,
                                                             build_function=measured_based_functions.build_measure_GISFM,
                                                             **params)
    elif config["method"] == "Cd-isfm":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.DynamicMeasureBasedFunction(ensemble_matrix, target,
                                                                    dynamic_measure_function=measured_based_functions.dynamic_iisfm,
                                                                    integral="choquet", **params)
    elif config["method"] == "Cd-mhm":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.DynamicMeasureBasedFunction(ensemble_matrix, target,
                                                                    dynamic_measure_function=measured_based_functions.dynamic_mhm,
                                                                    integral="choquet", **params)
    elif config["method"] == "Cg-mhm":
        # Get params
        params = get_choquet_params(config)
        return measured_based_functions.ChoquetBasedFunction(ensemble_matrix, target,
                                                             build_function=measured_based_functions.build_global_mhm,
                                                             **params)
    elif config["method"] == "Ceb":
        return measured_based_functions.EntropyMeasureBasedFunction(ensemble_matrix, target, integral="choquet")

    elif config["method"] == "Sm":
        # Get params
        params = get_sugeno_params(config)
        return measured_based_functions.SugenoBasedFunction(ensemble_matrix, target,
                                                            build_function=measured_based_functions.build_measure,
                                                            **params)
    elif config["method"] == "Sm+":
        # Get params
        params = get_sugeno_params(config)
        return measured_based_functions.SugenoBasedFunction(ensemble_matrix, target,
                                                            build_function=measured_based_functions.build_measure_additive,
                                                            **params)
    elif config["method"] == "Sml":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.LambdaFuzzyIntegralFunction(ensemble_matrix, target, integral="sugeno",
                                                                    **params)
    elif config["method"] == "Soifm":
        # Get params
        params = get_sugeno_params(config)
        return measured_based_functions.SugenoBasedFunction(ensemble_matrix, target,
                                                            build_function=measured_based_functions.build_measure_m_squared_mean_aggregation,
                                                            **params)
    elif config["method"] == "Sg-isfm":
        # Get params
        params = get_sugeno_params(config)
        return measured_based_functions.SugenoBasedFunction(ensemble_matrix, target,
                                                            build_function=measured_based_functions.build_measure_GISFM,
                                                            **params)
    elif config["method"] == "Sd-isfm":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.DynamicMeasureBasedFunction(ensemble_matrix, target,
                                                                    dynamic_measure_function=measured_based_functions.dynamic_iisfm,
                                                                    integral="sugeno", **params)
    elif config["method"] == "Sd-mhm":
        # Get optional parameters
        params = {}
        if "score" in config:
            params["score"] = config["score"]
        return measured_based_functions.DynamicMeasureBasedFunction(ensemble_matrix, target,
                                                                    dynamic_measure_function=measured_based_functions.dynamic_mhm,
                                                                    integral="sugeno", **params)
    elif config["method"] == "Sg-mhm":
        # Get params
        params = get_sugeno_params(config)
        return measured_based_functions.SugenoBasedFunction(ensemble_matrix, target,
                                                            build_function=measured_based_functions.build_global_mhm,
                                                            **params)
    elif config["method"] == "Seb":
        return measured_based_functions.EntropyMeasureBasedFunction(ensemble_matrix, target, integral="sugeno")
    else:
        raise Exception("'%s' method not recognized" % config["method"])
