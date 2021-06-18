import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from qiskit import IBMQ
import argparse

from .config import QKSConfig
from .frames_dataset import *
from .qks import QuantumKitchenSinks

def parse_args():
    parser = argparse.ArgumentParser(description='Quantum Kitchen Sink training script')
    parser.add_argument('--n-episodes', type=int, default=20, 
        help='Number of episodes to run QKS.')
    parser.add_argument('--scale', type=int, default=1, 
        help='Standard deviation of QKS\'s normal distribution.')
    parser.add_argument('--distribution', type=str, default='normal', 
        help="The distribution to use for QKS values.")
    parser.add_argument('--n-trials', type=int, default=1000, 
        help='Number of trials to run on each QuantumCircuit')
    parser.add_argument('--qubits', type=int, default=4, 
        help='Number of qubits.')
    parser.add_argument('--img-dir', type=str, default='fig/', 
        help='Directory to save figures')
    parser.add_argument('--no-plot', action='store_true', 
        help='Whether to plot the performance on the test and train datasets.')
    parser.add_argument('--tiling', action='store_true', 
        help='Whether to run QKS with tiling,')
    parser.add_argument('--no-cuda', action='store_true', 
        help='This flag should be passed when running the training script on a machine that does '
             'not support cuda. [NOT USED YET]')
    args = parser.parse_args()
    return args

def logistic_regression(train_dataset, test_dataset):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_dataset.X, train_dataset.y)

    train_acc = lr.score(train_dataset.X, train_dataset.y)
    test_acc = lr.score(test_dataset.X, test_dataset.y)

    print(
        "\naccuracy\n----- \n training: {}\n test:     {}\n"
          .format(train_acc, test_acc)
         )

    train_preds = lr.predict(train_dataset.X)
    test_preds = lr.predict(test_dataset.X)

    return train_preds, test_preds

def load_IBMQ_account():
    # Write the API token to IBM Q
    my_api_token = 'f84c8a32633b1ce7fda84ebc53c33141d96526678b3eba6ed5709619cb2d7378a0770a72831f01ded'\
        '4dd4a9ce691cfc5042e3c622b03129ce5376c5f000b5dcf'
    IBMQ.save_account(my_api_token)

    # Check the connection for IBM Q
    try:
        IBMQ.load_account(overwrite=True)
    except:
        print("""WARNING: There's no connection with the API for remote backends.
                Have you initialized a file with your personal token?
                For now, there's only access to local simulator backends...""")

def main():
    args = parse_args()

    print('Loading IBMQ account...')
    load_IBMQ_account()

    config = QKSConfig(qubits=args.qubits,
                       n_episodes=args.n_episodes,
                       scale=args.scale,
                       distribution=args.distribution,
                       n_trials=args.n_trials,
                       tiling=args.tiling,
                       cuda=not args.no_cuda)

    print('Initializing datasets...')
    QKS = QuantumKitchenSinks(config)
    train_dataset = QKSFramesDataset(QKS, is_train=True)
    test_dataset = QKSFramesDataset(QKS, is_train=False)

    print('Running logistic regression...')
    train_preds, test_preds = logistic_regression(train_dataset, test_dataset)

    if not args.no_plot:
        print('Making plots...')
        if not os.path.exists(args.img_dir):
            os.mkdir(args.img_dir)
        train_dataset.plot_results(train_preds, args.img_dir)
        test_dataset.plot_results(test_preds, args.img_dir)
        
    print('Done.')

if __name__ == "__main__":
    np.random.seed(1337)
    main()
