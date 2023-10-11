"""
The substrate that is optimized by the CPPN.
Taken, modified and improved from the Pure Python Library for ES-HyperNeat (pureples)
"""

class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates, hidden coordinates and a resolution defaulting to 10.0.
    Input and output coordinates are automatically calculated based on the number of inputs and outputs.
    There is one hidden layer by default.
    """

    def __init__(self, num_inputs, num_outputs, n_hidden_nodes=1, hidden_nodes=10, res=10.0):
        """ Initializes a Substrate object. 
        Args:
            num_inputs (_type_): The number of inputs used by the substrate
            num_outputs (_type_): _description_
            hidden_nodes (int, optional): _description_. Defaults to 10.
            res (float, optional): _description_. Defaults to 10.0.
        """
        self.input_coordinates = [(float(i), -1.0) for i in range(-num_inputs//2, num_inputs//2)]
        self.output_coordinates = [(float(i), 1.0) for i in range(-num_outputs//2, num_outputs//2)]
        self.hidden_coordinates = [[(float(i), 0.0) for i in range(-hidden_nodes//2, hidden_nodes//2)] for _ in range(n_hidden_nodes)]
        self.res = res