import numpy as np


def sugeno_elements(x, ds):
    """
    Function to calculate sugeno integral values
    :param x: Array with values
    :param ds: DataStore with the measure values
    :return: the sugeno integral values
    """
    # Create a list from [1, len(x)]
    T = list(range(1, len(x)+1))
    # Sort the values and match each index with the value
    ux = sorted(zip(x, T), key=lambda y:y[0])
    # Create an empty list
    S = []
    # Fore each element
    for i in range(len(x)):
        # Get the value to aggregate and the measure value
        S.append((ux[i][0], ds.get_no_transform(tuple(T))))
        # Remove from T used element
        T.remove(ux[i][1])

    # Return the sugeno elements
    return np.array(S)
