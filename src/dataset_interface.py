import abc
import numpy as np

from torch.utils.data import Dataset, DataLoader

from .qks import QuantumKitchenSinks


class QKSDatasetInterface(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, QKS: QuantumKitchenSinks, is_train=True):
        super(QKSDatasetInterface, self).__init__()

        classical_dataset = self._get_classical_dataset(is_train)
        self.X = QKS.qks_preprocess(classical_dataset, self._vectorize_input, self._get_input_shape())
        self.y = self._get_classical_dataset_labels()
    
    def __getitem__(self, index):
        """
        Gets an item of the Dataset

        return: Tuple[np.array, float]
        """
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    @abc.abstractmethod
    def plot_results(self, train_preds, test_preds, img_dir: str):
        """
        Plot the results of training and save figures. 

        param train_preds: training predictions
        param test_preds: test predictions
        param img_dir: str directory to save the images
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _vectorize_input(self, x) -> np.array:
        """
        Turn an element of the classical dataset into a flat numpy float array

        return: int
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _get_input_shape(self) -> int:
        """
        Get the size of each input vector in the classical dataset.

        return: int
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _get_classical_dataset_labels(self) -> np.array:
        """
        Get np.array of labels in the classical dataset

        return: np.array labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_classical_dataset(self, is_train: bool) -> Dataset:
        """ 
        Get classical dataloader to be transformed. Batch size should be 1.

        param is_train: bool indicating if the Dataset is training data
        return: torch.utils.data.Dataset, each element should be a Tuple[Input, Label]
        """
        raise NotImplementedError
    