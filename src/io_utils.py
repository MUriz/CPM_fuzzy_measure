from collections import OrderedDict
import numpy as np


def read_confidence_matrix(filename):
    """
    Function that reads a file of confidence matrix.
    Each line of the file are the classifier outputs (confidences) separated by tab (\t)
    :param filename: the name of the file
    :return: a numpy matrix of n_classifiers rows by n_instances columns
    """
    # Create an empty list to store confidence matrix
    confidence_matrix = []
    # Open file
    with open(filename, 'r') as f:
        # Read the first line
        l = f.readline()
        # While there are more lines
        while l.strip() != '':
            # Split line by tab (\t) and parse to float
            values = list(map(lambda x: float(x.strip()), l.strip().split('\t')))
            # Append values to list
            confidence_matrix.append(values)
            # Go to next line
            l = f.readline()
    # Return the read values but in numpy array
    return np.array(confidence_matrix)


def read_out_matrix(filename):
    """
    Function that reads a file of prediction of each classifier for each instance
    Each line of the file are the classifier outputs (classes) separated by tab (\t)
    :param filename: the name of the file
    :return: a numpy matrix of n_classifiers rows by n_instances columns
    """
    # Create an empty list to store output matrix
    out_matrix = []
    # Open file
    with open(filename, 'r') as f:
        # Read the first line
        l = f.readline()
        # While there are more lines
        while l.strip() != '':
            # Split line by tab (\t) and parse to int
            values = list(map(lambda x: int(x.strip()), l.strip().split('\t')))
            # Append values to list
            out_matrix.append(values)
            # Go to next line
            l = f.readline()
    # Return the read values but in numpy array
    return np.array(out_matrix)


class Dataset(object):
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = []
        self.target_names = []
        self.categorical = []


def read_keel_file(file_path):
    """
    A function that reads a keel data file
    :param file_path: the file name
    :return: Readedata and target
    """
    # Open file
    f = open(file_path, 'r')
    # Read the first line
    line = f.readline()
    # Create an ordered dict
    att = OrderedDict()
    # Create an object to store data
    ds = Dataset()
    # Default class name
    class_name = 'Class'
    # While line is not in @data
    while line.find("@data") < 0:
        # If the line is an @attribute and it is real or integer
        if line.find("@attribute") >= 0 and ((line.find("real") >= 0) or (line.find("integer") >= 0)):
            # Get the lower bound
            min_val = line.split("[")
            min_val = float(min_val[1].split(",")[0])
            # Get the upper bound
            max_val = float(line.split("]")[0].split(',')[1])
            # Create a dictionary with the name of the attribute as key and the bounds as value
            att_aux = {line.split()[1].split('[')[0]: [min_val, max_val]}
            # Update the att dictionary with the attribute information
            att.update(att_aux)
            # And set categorical as False
            ds.categorical.append(False)
        # If the line is an @attribute and it is not real (it have to be categorical)
        elif line.find("@attribute") >= 0 and line.find("real") < 0:
            # List to store values
            values = []
            # Split line by "{"
            l = line.split('{')
            # Append the first value
            values.append(l[1].split(',')[0].strip())
            # Split by ","
            line2 = line.split(',')
            # For the second value to last (not include)
            for l in line2[1:-1]:
                # Append the value
                values.append(l.strip())
            # Get the las value
            l = line2[-1]
            # Remove "}"
            l = l.split('}')[0]
            # Append to values
            values.append(l.strip())
            # Create the attribute dict (name as key and values as value)
            att_aux = {line.split()[1].split('{')[0]: values}
            # Update the attribute dictionary
            att.update(att_aux)
            # Set categorical s true
            ds.categorical.append(True)
        # If the line is @output or @outputs
        elif line.find("@output") >= 0 or line.find("@outputs") >= 0:
            # Get the class name
            class_name = line.split()
            class_name = class_name[1]
        # Go to next line
        line = f.readline()

    # Get the class values
    aux_class_values = att.pop(class_name)
    # Remove the last element from categorical list (the class attribute)
    ds.categorical = ds.categorical[:-1]
    # Class values
    class_values = aux_class_values[:]
    # Create the class attribute with the class values
    att_aux = {class_name: class_values}
    # Add to dictionary
    att.update(att_aux)
    # Go to next line (data line)
    line = f.readline()
    # List to store the classes of the examples, the examples data and
    exp_classes = []
    examples = []
    exp_original = []
    # While there is more content
    while line != "":
        # Replace "," for " "
        line = line.replace(",", " ")
        # Split line
        l = line.split()
        # Get values (not class)
        values = l[0:len(l) - 1]
        # Create empty list
        val = []
        val_original = []
        # For each value
        for i, v in enumerate(values):
            ####
            # Missing value, quit
            if v == "?":
                break
            else:
                # If the ith attribute is categorical
                if ds.categorical[i]:
                    # Get key list
                    key_list = list(att)
                    # Get the values of the ith key
                    att_values = att[key_list[i]]
                    # Append the index of the value in list
                    val.append(att_values.index(v))
                    # Append original value
                    val_original.append(v)
                else:
                    # It is not categorical so it can be integer or float
                    # Append the float value to list
                    val.append(float(v))
                    val_original.append(float(v))
        # If the value is not missing value
        if v != "?":
            # Append to examples the values
            examples.append(val)
            # The original too
            exp_original.append(val_original)
            # Get the class values
            att_values = att[class_name]
            # Append to classes the indice of the class (not value)
            exp_classes.append(att_values.index(l[len(l) - 1]))
        # Go to next line
        line = f.readline()
    # Convert examples to numpy array
    examples = np.array(examples)
    # Close file
    f.close()
    # Update data in Datset object
    ds.data = examples
    # Update target
    ds.target = np.array(exp_classes)
    # Get the attribute names
    aux = list(att)
    # Update feature names and class values
    ds.feature_names = aux[:-1]
    ds.target_names = att[class_name]
    # Return Dataset object and the original read values
    return ds, exp_original


def read_matrix_and_target(data_file, confidence_file, out_file):
    """
    Function that reads data, confidence and output file
    :param data_file: data file in keel format
    :param confidence_file: confidence file, where each line are the classifier confidence values separated by tab (\t)
    :param out_file: output file, where each line are the classifier class values separated by tab (\t)
    :return: a new confidence matrix (num_classifiers, num_classes, num_instances) and an array with the true class
    values
    """
    # Read confidence matrix and output
    confidence_matrix = read_confidence_matrix(confidence_file)
    out_matrix = read_out_matrix(out_file)
    # Get the number of classifiers and instances
    n_classifiers, n_instances = confidence_matrix.shape
    # Create a matrix of num_classifiers, num_classes, num_examples
    # We work with two classes
    M = np.empty((n_classifiers, 2, n_instances))
    # For each instance
    for i in range(n_instances):
        # For each classifier
        for c in range(n_classifiers):
            # Get the output
            out = out_matrix[c, i]
            # If class is 0
            if out == 0:
                # The confidence of class 0 is the read value
                M[c, 0, i] = confidence_matrix[c, i]
                # The confidence for class 1 is 1 minus the read value
                M[c, 1, i] = 1 - confidence_matrix[c, i]
            else:
                # If the class is 1
                # The confidence of class 0 is 1 minus read value
                M[c, 0, i] = 1 - confidence_matrix[c, i]
                # The confidence of class 1 is the read value
                M[c, 1, i] = confidence_matrix[c, i]

    # Finally read data file
    data, _ = read_keel_file(data_file)
    # Return the new confidence matrix and the target of the data (real class values)
    return M, data.target


def read_config_file(file_path):
    """
    Reads the config file, where each line is configuration parameter with the format key=value
    The lines started by "#" will be ignored taking them as comments
    :param file_path: The path of the config file
    :return: A dictionary with read configuration
    """
    config = dict()
    with open(file_path, 'r') as f:
        for l in f:
            if l.strip().startswith("#") or l == "\n":
                # Skip line
                continue
            # Split line by "="
            key, value = l.strip().split("=")
            # Add to config dict
            config[key.strip()] = value.strip()
    return config
