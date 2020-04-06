import numpy as np


def choquet_elements(x, ds):
    """
    Function to calculate choquet integral
    :param x: Array with values
    :param ds: DataStore with the measure values
    :return: the choquet integral values
    """
    # Create a list from [1, len(x)]
    T = list(range(1, len(x)+1))
    # Sort the values and match each index with the value
    ux = sorted(zip(x, T), key=lambda y: y[0])
    # Start a list with the first element and the measure value associated to it
    S = [(ux[0][0], ds.get_no_transform(tuple(T)))]
    # Remove from T the used element
    T.remove(ux[0][1])
    # For the rest of values
    for i in range(1, len(x)):
        # Get the new value (the current minus the previous) and the measure value associated to it
        S.append(((ux[i][0] - ux[i-1][0]), ds.get_no_transform(tuple(T))))
        # Remove from T used element
        T.remove(ux[i][1])

    # Return the choquet elements
    return np.array(S)
