import abc

from typing import Iterable, List, Tuple
from torch.utils.data import Dataset

from .qks import QuantumKitchenSinks


class QKSDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self):
        super.__init__()

        X_train, X_test, y_train, y_test = self._get_classical_dataset()
        
    
    def get_item(index):
        item = {

        }
        return item

    @abc.abstractmethod
    def _get_classical_dataset(self) -> Tuple[Iterable, Iterable, Iterable, Iterable]:
        """ 
        Get test and train inputs and labels.

        return: Tuple[Iterable, Iterable, Iterable, Iterable] X_train, X_test, y_train, y_test
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _clean_input(self, feature) -> List[float]:
        """
        Takes an input feature and returns it in a format QKS recognizes. Meant to turn features
        wrapped in things like torch.tensor(...) into python lists.

        param feature: An element of train/test dataset.
        return: List[float] cleansed version of the input feature
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _clean_label(self, feature):
        """
        Takes a label and returns it in a format QKS recognizes. Meant to turn features
        wrapped in things like torch.tensor(...) into python lists.

        param feature: A label of the train/test dataset.
        return: cleansed version of the input feature
        """
        raise NotImplementedError