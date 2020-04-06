import itertools


class DictDataStore(object):
    """
    Class to store the measure values with a python dictionary
    """

    def __init__(self, n):
        """
        Constructor
        :param n: The number of elements
        """
        # Store the number of elements
        self.n = n
        # Create dicts to store data and keys
        self.data = dict()
        self.keys = dict()
        # Initialize current key with 1
        self.cur_key = 1

    def clone(self):
        """
        Return a new DictDataStore with the same values
        :return: cloned DictDataStore
        """
        # Create the object
        r = DictDataStore(self.n)
        # Copy data
        r.data = dict(self.data)
        r.keys = dict(self.keys)
        r.cur_key = self.cur_key
        # Return it
        return r

    def put(self, key, value):
        """
        Method to put a value on key
        :param key: The key
        :param value: The value
        :return: Nothing
        """
        # We use a choquet implementation that needs the keys to be in [1, N],
        # so we transform indices to do so
        # To store new transformed key
        new_key = []
        # Transform key
        for i in key:
            # If not transformed yet, associate new index
            if not i in self.keys:
                self.keys[i] = self.cur_key
                self.cur_key += 1
            new_key.append(self.keys[i])
        # Add the element to data
        new_key = tuple(new_key)
        self.data[new_key] = value

    def get(self, key):
        """
        Method to get the element associated with the key
        :param key: The key used to store (given by user not transformed)
        :return: The associated value
        """
        # Transform the key.
        new_key = []
        for i in key:
            new_key.append(self.keys[i])
        # Return the value of the key
        new_key = tuple(new_key)
        return self.data[new_key]

    def get_no_transform(self, key):
        """
        Method to get the associated value without transforming the given key
        :param key: The key
        :return: The associated value
        """
        return self.data[key]

    def correct_monotonicity(self):
        """
        Function to correct monotonicity
        The parents of elements must be at least equal to the bigger of its children
        Here we update the parent values to be at least equal to the bigger of its children
        :return:
        """
        # For each level (2 index or more)
        for i in range(1, self.n):
            # Get the keys with (i+1) elements
            kys = filter(lambda x: len(x) == i + 1, self.data.keys())
            # For each key
            for k in kys:
                # Set mx to the current node
                mx = self.data[k]
                # For all possible child
                for j in itertools.combinations(k, i):
                    # Update the max value
                    mx = max(self.data[j], mx)
                # Update current node to max value
                self.data[k] = mx

    def correct_monotonicity_top_down(self):
        """
        Method to correct monotonicity
        Here the children of the parent are update when the children value is bigger than the parent value
        :return:
        """
        for i in range(self.n - 1, 0, -1):
            kys = filter(lambda x: len(x) == i, self.data.keys())
            for k in kys:
                mn = self.data[k]
                s1 = set(k)
                for j in itertools.combinations(range(1, self.n + 1), i + 1):
                    s2 = set(j)
                    if s1.issubset(s2):
                        if self.data[j] < mn:
                            mn = self.data[j]
                self.data[k] = mn

    '''
    Method to normalize data
    The top node (the node with key of all elements) must be 1.0
    We divide all values by top node value
    '''

    def normalize(self):
        """
        Method for normalize (set 1 the "top" node and scale others nodes too)
        :return:
        """
        # Get the "top" node
        top_key = list(filter(lambda x: len(x) == self.n, self.data.keys()))[0]
        # divide each value with the top node value
        mx_value = self.data[top_key]
        if mx_value == 0.0:
            self.data[top_key] = 1.0
        else:
            for i in self.data.keys():
                self.data[i] /= float(mx_value)

    def set_top_to_1(self):
        """
        Method that sets to 1 the "top" node value, without changing any other node
        :return:
        """
        top_key = filter(lambda x: len(x) == self.n, self.data.keys())[0]
        self.data[top_key] = 1.0

    def scale(self, alpha):
        """
        Method to multiply each data store "level" by alpha^(num_classifiers-num_total)
        :param alpha: Scale value
        :return:
        """
        for key in self.data.keys():
            self.data[key] *= (alpha ** (self.n - len(key)))

    '''
    Method to scale values linearly
    The TOP node will be multiplied by 1 and the lower nodes (no 0 node) by min_alpha
    The between nodes will be multiplied in linear form between min_alpha and 1.0
    @param min_alpha the min alpha value (lower nodes multiplier)
    '''

    def scale2(self, min_alpha):
        """
        Method to scale values linearly
        The "top" node will be multiplied by 1 and the lower nodes (no 0 node) by min_alpha
        The between nodes will be multiplied in linear form between min_alpha and 1.0
        :param min_alpha: The min alpha value
        :return:
        """
        # Formula:
        # alpha(i) = (min_alha * (num_classifiers - i) + i - 1) / (num_classifiers - 1)
        for key in self.data.keys():
            a = float((min_alpha * (self.n - len(key)) + len(key) - 1.0))
            b = float(self.n - 1.0)
            alpha = a / b
            self.data[key] *= alpha

    def multiply_by(self, value):
        """
        Method that multiply each element by a given value
        :param value: The value to multiply by
        :return:
        """
        for key in self.data.keys():
            self.data[key] *= value

    def __add__(self, other):
        """
        Method that adds two DatStores and returns the added one
        :param other: Other Data store
        :return: The added DataStore
        """
        ret_ds = self.clone()
        try:
            for i in ret_ds.data.keys():
                ret_ds.data[i] += other.data[i]
        except KeyError:
            raise Exception("The data stores don't have the same keys")

        return ret_ds

    def __eq__(self, other):
        """
        Method to check if two DataStore are equal
        :param other: The other data store
        :return: True if all elements are equal, False otherwise
        """
        try:
            for j in self.data.keys():
                if self.data[j] != other.data[j]:
                    return False
            return True
        except:
            return False
