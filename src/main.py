import argparse
import os
import io_utils
import config
import numpy as np


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", action="store", required=True, dest="config_file",
                        help="The configuration file")
    return parser


if __name__ == '__main__':
    arguments = get_argument_parser().parse_args()
    # Verify that the file exists
    if not os.path.isfile(arguments.config_file):
        raise Exception("The configuration file does not exist or is not a file")

    # Read config
    conf = io_utils.read_config_file(arguments.config_file)

    # Get the aggregator
    aggregator = config.get_aggregator(conf)

    # Read train and test data
    train_M, train_target = io_utils.read_matrix_and_target(conf["train_data"], conf["train_conf"], conf["train_out"])
    test_M, test_target = io_utils.read_matrix_and_target(conf["test_data"], conf["test_conf"], conf["test_out"])

    # Check that the number of classes and the number of classifiers are equal for train and test
    if train_M.shape[0] != test_M.shape[0]:
        raise Exception("Train and test have different number of classifiers")

    if train_M.shape[1] != test_M.shape[1]:
        raise Exception("Train and test have different number of classes")

    # Get the number of classifiers
    num_classifiers = int(conf["num_classifiers"])

    # Check that the number of classifiers is lower than the total num of classifiers 
    if num_classifiers > train_M.shape[0]:
        raise Exception("There are only %d classifiers in data and %d are given" % (train_M.shape[0], num_classifiers))

    # Get only the first num_classifiers
    train_M = train_M[:num_classifiers, :, :]
    test_M = test_M[:num_classifiers, :, :]

    # Aggregate
    train_agg = aggregator.aggregate(train_M)
    test_agg = aggregator.aggregate(test_M)

    # Predict getting class with bigger aggregated confidence
    train_prediction = np.argmax(train_agg, axis=1)
    test_prediction = np.argmax(test_agg, axis=1)

    # Write a csv wth the target and prediction
    train_csv_out = "target,prediction\n"
    for i in range(train_prediction.shape[0]):
        train_csv_out += "%d,%d\n" % (train_target[i], train_prediction[i])

    test_csv_out = "target,prediction\n"
    for i in range(test_prediction.shape[0]):
        test_csv_out += "%d,%d\n" % (test_target[i], test_prediction[i])

    with open(conf["train_pred"], 'w') as f:
        f.write(train_csv_out)

    with open(conf["test_pred"], 'w') as f:
        f.write(test_csv_out)
