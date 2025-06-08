import sys
# Assuming torch_train.py is in the same directory
from .torch_train import main, parse_args

if __name__ == '__main__':
    # Simulate command line arguments for Cora
    sys.argv = sys.argv[:1] + [
        '--dataset', 'cora',
        '--epochs', '10', # Override for a quick test run
        '--early_stopping', '3', # Override for a quick test run
        # Other parameters will use defaults from torch_train.parse_args()
        # Example: to use specific values from original TF flags for SGLCN on Cora:
        # '--lr1', '0.005',
        # '--lr2', '0.005',
        # '--hidden_gl', '70',
        # '--hidden_gcn', '30',
        # '--losslr1', '0.01',
        # '--losslr2', '0.0001',
        # '--weight_decay', '1e-4', # Make sure this matches how it's used in torch_train
        # '--dropout', '0.6',
    ]
    args = parse_args()
    print("Running Cora with settings:")
    print(args)
    main(args)
