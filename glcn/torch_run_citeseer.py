import sys
# Assuming torch_train.py is in the same directory
from .torch_train import main, parse_args

if __name__ == '__main__':
    # Simulate command line arguments for Citeseer
    sys.argv = sys.argv[:1] + [
        '--dataset', 'citeseer',
        '--epochs', '10', # Override for a quick test run
        '--early_stopping', '3', # Override for a quick test run
        # Other parameters will use defaults from torch_train.parse_args()
        # Example: to use specific values:
        # '--lr1', '0.005',
        # '--lr2', '0.005',
        # '--losslr1', '0.01', # These might need tuning for Citeseer
        # '--losslr2', '0.0001',
    ]
    args = parse_args()
    print("Running Citeseer with settings:")
    print(args)
    main(args)
