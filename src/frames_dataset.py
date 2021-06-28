import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from .dataset_interface import QKSDatasetInterface
from .qks import QuantumKitchenSinks


def get_frames_dataloaders(QKS: QuantumKitchenSinks, batch_size=32):
    train = QKSFramesDataset(QKS, is_train=True)
    test = QKSFramesDataset(QKS, is_train=False)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train, test, 1


class QKSFramesDataset(QKSDatasetInterface):
    def __init__(self, QKS: QuantumKitchenSinks, is_train: bool = True):
        self.is_train = is_train
        super(QKSFramesDataset, self).__init__(QKS, is_train)

    @staticmethod
    def _make_plot(data, target, name, title):
        plt.figure(figsize=(5, 5))
        plt.title(title)
        plt.scatter(data[:,0], data[:,1], s=5, c=target)
        plt.savefig(name)

    def plot_results(self, preds, img_dir):
        name = 'training_dataset' if self.is_train else 'test_dataset'
        title = 'Training Set' if self.is_train else 'Test Set'
        self._make_plot(self.classical_dataset.X, self.y, os.path.join(img_dir, name), title)

        name = 'results_experiment_train' if self.is_train else 'results_experiment_test'
        title = 'Training Set' if self.is_train else 'Test Set'
        self._make_plot(self.classical_dataset.X, preds, os.path.join(img_dir, name), title)
    
    def _vectorize_input(self, x) -> np.array:
        return np.array(x.squeeze().tolist())
    
    def _get_input_shape(self) -> int:
        return self.classical_dataset.X[0].shape[0]
    
    def _get_classical_dataset_labels(self) -> np.array:
        return self.classical_dataset.y

    def _get_classical_dataset(self, is_train: bool) -> Dataset:
        if is_train:
            self.classical_dataset = FramesDataset(200, 2, 1)
        else:
            self.classical_dataset = FramesDataset(50, 2, 1)
        return self.classical_dataset

class FramesDataset(Dataset):
    """ Generates pictures frames dataset. """
    def __init__(self, size: int, outer_length: int, inner_length: int):
        super(FramesDataset, self).__init__()

        X, y = self._make_data(size, outer_length, inner_length)

        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def _zip_arrays(self, d1, d2):
        return np.array(list(zip(d1, d2)))

    def _shuffle(self, d1, d2):
        assert len(d1) == len(d2)
        p = np.random.permutation(len(d1))
        return d1[p], d2[p]

    def _make_barriers(self, num, length):
        dist = 0.1 * length
        left = np.random.uniform((-length - dist), (-length + dist), num)
        right = np.random.uniform((length - dist), (length + dist), num)
        left_barrier = np.random.uniform(-length, length, num)
        right_barrier = np.random.uniform(-length, length, num)
        L = self._zip_arrays(left, left_barrier)
        R = self._zip_arrays(right, right_barrier)
        barriers = np.vstack((L, R))
        return barriers

    def _make_squares(self, num, length):
        left_right, top_bottom = self._make_barriers(num, length), self._make_barriers(num, length)
        top_bottom[:, [0, 1]] = top_bottom[:, [1, 0]]  # swaps X and Y axis
        square = np.concatenate((left_right, top_bottom))
        return square

    def _make_data(self, size, outer_length, inner_length):
        outer = self._make_squares(size, outer_length)
        inner = self._make_squares(size, inner_length)
        assert len(outer) == len(inner)

        frames = np.concatenate((outer, inner))
        target = np.concatenate((np.zeros(len(outer)), np.ones(len(inner))))

        X, y = self._shuffle(frames, target)
        return X, y
