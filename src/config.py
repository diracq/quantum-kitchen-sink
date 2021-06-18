class QKSConfig:
    def __init__(self,
                 qubits: int = 4,
                 n_episodes: int = 20,
                 scale: int = 1,
                 distribution: str = 'normal',
                 n_trials: int = 1000,
                 tiling: bool = False, 
                 cuda: bool = True):
        """
        param qubits: int number of qubits to use
        param n_episodes: int number of episodes the dataset is iterated over (i.e. for each sample 
                          in the dataset, we apply the QKS transformation n_episodes number of times)
        param scale: int standard deviation (spread) of the normal distribution
        param distribution : string distribution used to create nonzero elements of omega
        param n_trials: int the number of times we run the quantujm circuit
        param tiling: bool whether or not to tile the input
        param cuda: bool enable cuda for torch
        """
        self.qubits = qubits
        self.n_episodes = n_episodes
        self.scale = scale
        self.distribution = distribution
        self.n_trials = n_trials
        self.tiling = tiling
        self.cuda = cuda
        