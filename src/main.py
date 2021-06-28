import numpy as np

from comet_ml import Experiment

from .config import QKSConfig
from .frames_dataset import *
from .qks import QuantumKitchenSinks
from .mnist_dataset import get_mnist_dataloaders
from .model import Net
from .utils import *


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
    train_loader, test_loader, _, _, output_classes = get_mnist_dataloaders(QKS)

    print('Initializing model...')
    model = Net(config.qubits*config.n_episodes, output_classes).to(device)

    experiment = Experiment(log_code=False, disabled=args.debug)

    print('Training...')
    for e in range(args.epochs):
        train(experiment, model, train_loader)
        test(experiment, model, test_loader, e)

    print('Saving model...')
    save(model, './model.pt')
        
    print('Done.')

if __name__ == "__main__":
    np.random.seed(1337)
    main()
