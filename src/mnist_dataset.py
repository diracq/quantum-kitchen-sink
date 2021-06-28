import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from .dataset_interface import QKSDatasetInterface
from .qks import QuantumKitchenSinks


def get_mnist_dataloaders(QKS: QuantumKitchenSinks, batch_size=32):
    train = QKSMnistDataset(QKS, is_train=True)
    test = QKSMnistDataset(QKS, is_train=False)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train, test, 10


class QKSMnistDataset(QKSDatasetInterface):
    def __init__(self, QKS: QuantumKitchenSinks, is_train: bool = True):
        self.is_train = is_train
        super(QKSMnistDataset, self).__init__(QKS, is_train)

    def plot_results(self, preds, img_dir):
        print('MNIST doesn\'t support plotting')
    
    def _vectorize_input(self, x) -> np.array:
        return np.array(x).reshape(-1)
    
    def _get_input_shape(self) -> int:
        return 28**2
    
    def _get_classical_dataset_labels(self) -> np.array:
        return self.mnist.targets

    def _get_classical_dataset(self, is_train: bool) -> Dataset:
        if is_train:
            self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        else:
            self.mnist = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        return self.mnist
