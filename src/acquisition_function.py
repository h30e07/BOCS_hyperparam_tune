class AcquisitionFunction():
    def __init__(self, num_reads=20, num_sweeps=1000):
        """
        Class of the acquisition function for BOCS.
        Args:
            num_reads (int): Number of sampling reads
            num_sweeps (int): Number of sweeps for each sampling

        """
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps

    def build(self, bbo_params):
        """
        Build the acquisition function with the given surrogate model parameters.
        Args:
            bbo_params (numpy.ndarray): Parameters of the surrogate model
        
        """
        pass

    def optimize(self):
        """
        Find the next point to evaluate by optimizing the acquisition function.
        Returns:
            next_x (numpy.ndarray): The next point to evaluate of shape (1, D)
        """
        next_x = None
        return next_x