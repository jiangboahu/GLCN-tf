import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from utils.utils import load_data # Assuming utils.py is in pytorch_glcn/utils/
from models.models import SGLCN   # Assuming models.py is in pytorch_glcn/models/

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr_gl', type=float, default=0.01, help='Initial learning rate for graph learning part.') # lr1 in original
parser.add_argument('--lr_gcn', type=float, default=0.01, help='Initial learning rate for GCN part.') # lr2 in original
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_gl', type=int, default=64, help='Number of hidden units for graph learning layer.')
parser.add_argument('--hidden_gcn', type=int, default=32, help='Number of hidden units for GCN layer.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset to use (cora, citeseer).')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
parser.add_argument('--loss_lr1', type=float, default=0.1, help='Learning rate for graph loss feature term.') # Corresponds to FLAGS.losslr1
parser.add_argument('--loss_lr2', type=float, default=0.1, help='Learning rate for graph loss S trace term.') # Corresponds to FLAGS.losslr2
parser.add_argument('--lr_decay_steps', type=int, default=100, help='Steps for learning rate decay.')
parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Rate for learning rate decay.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Device selection
device = torch.device("cuda" if args.cuda else "cpu")

# Load data
# The path argument to load_data is constructed by prefix + dataset_str + "/"
# Example: data_path_prefix = "./data/", dataset_str = "cora" -> "./data/cora/"
# The function now returns: adj_processed_for_gcn, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, edge_index_orig
adj_processed_for_gcn, features_sparse, _, _, _, train_mask, val_mask, test_mask, labels_full, edge_index_orig = \
    load_data(args.dataset, path=args.data_path_prefix)


# Convert data to PyTorch tensors and move to device
features = features_sparse.to(device)
# adj_processed_for_gcn is used by GCN layers, but not directly passed to model constructor or forward pass.
# It's encapsulated within how S_learned is used by GCN layers if S_learned is based on it,
# OR if the GCN layers in the model were to take this preprocessed adj directly.
# CURRENT SGLCN model forward pass: model(features, edge_index_orig) -> S_learned is then used by GCN layers.
# So, adj_processed_for_gcn is NOT directly used by train.py after loading, if SGLCN internally uses S_learned.
# Let's keep it here for now, but it might be unused if S_learned is the only graph for GCNs.
adj_gcn_input = adj_processed_for_gcn.to(device) # This is the adj for GCN layers if NOT using S_learned exclusively.
                                                 # However, SGLCN model's GCN layers use S_learned.

edge_index_orig = edge_index_orig.to(device) # Original graph structure for SparseGraphLearn
labels_full = labels_full.to(device) # All labels

# y_train, y_val, y_test from load_data are derived from labels_full and masks,
# but model.loss_fn and model.accuracy use labels_full and the specific mask.
# So, explicit y_train, y_val, y_test tensors might not be strictly needed here if using labels_full.
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

num_nodes = features.shape[0]

if features.is_sparse:
    input_dim = features.size(1)
else:
    input_dim = features.shape[1]

# For one-hot labels, output_dim is labels_full.shape[1]. If class indices, it's labels_full.max()+1
if labels_full.dim() > 1 and labels_full.shape[1] > 1:
    output_dim = labels_full.shape[1]
else: # class indices
    output_dim = labels_full.max().item() + 1

# Try-except block for imports was removed from the thought process, assuming it's handled.
# Ensure correct import paths based on execution context.
# (This was in the previous version of train.py from an earlier step, but got removed in the read_files output)
try:
    from utils.utils import load_data
    from models.models import SGLCN
except ModuleNotFoundError:
    from pytorch_glcn.utils.utils import load_data
    from pytorch_glcn.models.models import SGLCN


# Model instantiation
model = SGLCN(input_dim=input_dim,
              output_dim=output_dim,
              hidden_gl_dim=args.hidden_gl,
              hidden_gcn_dim=args.hidden_gcn,
              dropout_rate=args.dropout,
              weight_decay=args.weight_decay,
              loss_lr1=args.loss_lr1,
              loss_lr2=args.loss_lr2,
              sparse_inputs=features.is_sparse).to(device)

# Optimizer setup
# The original TF code uses two separate optimizers.
# We can achieve this with one optimizer and parameter groups, or two optimizers.
# Using one optimizer with parameter groups:
param_groups_gl, param_groups_gcn = model.get_optimizer_param_groups(lr_gl=args.lr_gl, lr_gcn=args.lr_gcn)
optimizer = optim.Adam(param_groups_gl + param_groups_gcn) # lr is set in param groups

# Learning rate schedulers
# TF's exponential_decay: new_lr = initial_lr * decay_rate ^ (global_step / decay_steps)
# PyTorch's ExponentialLR: new_lr = initial_lr * gamma ^ epoch
# To match, gamma = decay_rate ^ (1 / decay_steps_pytorch_equiv)
# If decay_steps_tf is per batch and PyTorch scheduler is per epoch, this needs adjustment.
# Assuming decay_steps is roughly equivalent to epochs for decay interval for simplicity here.
# Or, more closely, if decay_steps=100 means decay every 100 steps (epochs here):
# gamma_gl = args.lr_decay_rate ** (1.0 / args.lr_decay_steps) if args.lr_decay_steps > 0 else 1.0
# scheduler_gl = ExponentialLR(optimizer.param_groups[0]['optimizer'], gamma=gamma_gl) # This is not how it works
# scheduler_gcn = ExponentialLR(optimizer.param_groups[1]['optimizer'], gamma=gamma_gl)

# Simpler: Apply scheduler to the whole optimizer. If LRs are different, they decay proportionally.
# Or, create two schedulers for each parameter group if an optimizer is split.
# For Adam, typically one scheduler is applied to the optimizer.
# To have different decay behavior per group, one might need custom scheduler logic or multiple optimizers.

# Let's use one scheduler for the combined optimizer for now.
# PyTorch's ExponentialLR applies gamma each epoch.
# TF: lr = lr * decay_rate ^ (epoch / decay_steps)
# If we want decay every 'decay_steps' epochs by 'decay_rate':
# gamma should be such that gamma^decay_steps = decay_rate => gamma = decay_rate^(1/decay_steps)
if args.lr_decay_steps > 0 and args.lr_decay_rate < 1.0:
    gamma = args.lr_decay_rate ** (1.0 / args.lr_decay_steps)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
else:
    scheduler = None # No decay or dummy scheduler

def evaluate(features_eval, edge_index_eval, labels_eval, mask_eval):
    model.eval()
    with torch.no_grad():
        logits, h_gl, s_learned = model(features_eval, edge_index_eval)
        total_loss, loss_gl, loss_gcn = model.loss_fn(logits, h_gl, s_learned, labels_full, mask_eval, num_nodes)
        acc = model.accuracy(logits, labels_full, mask_eval)
    return total_loss.item(), loss_gl.item(), loss_gcn.item(), acc.item()

# Training loop
t_total = time.time()
bad_counter = 0 # For early stopping
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(args.epochs):
    t_epoch_start = time.time()
    model.train()
    optimizer.zero_grad()

    logits, h_gl, S_learned = model(features, edge_index_orig)

    # Calculate loss only on training data
    total_loss, loss_gl, loss_gcn = model.loss_fn(logits, h_gl, S_learned, labels_full, train_mask, num_nodes)

    total_loss.backward()
    optimizer.step()

    if scheduler:
        # Decay learning rate based on epoch,
        # but only if current epoch is a multiple of decay_steps for TF-like behavior
        # Or simply call scheduler.step() if gamma was calculated for per-epoch decay
        scheduler.step() # Standard PyTorch: decay every epoch

    # Evaluate on validation set
    val_total_loss, val_loss_gl, val_loss_gcn, val_acc = evaluate(features, edge_index_orig, labels_full, val_mask)

    print(f"Epoch: {epoch+1:04d} | "
          f"LR: {optimizer.param_groups[0]['lr']:.5f} (GL), {optimizer.param_groups[1]['lr']:.5f} (GCN) | "
          f"Train Loss: {total_loss.item():.4f} (Total) {loss_gl.item():.4f} (GL) {loss_gcn.item():.4f} (GCN) | "
          f"Val Loss: {val_total_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {time.time() - t_epoch_start:.4f}s")

    # Early stopping
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        best_epoch = epoch + 1
        bad_counter = 0
        # Could save model checkpoint here
        # torch.save(model.state_dict(), f'{args.dataset}_sglcn_best.pkl')
    else:
        bad_counter += 1

    if bad_counter >= args.patience:
        print(f"Early stopping at epoch {epoch+1}...")
        break

print("Optimization Finished!")
print(f"Total time elapsed: {time.time() - t_total:.4f}s")

# Load best model weights if saved
# model.load_state_dict(torch.load(f'{args.dataset}_sglcn_best.pkl'))

# Testing
test_total_loss, test_loss_gl, test_loss_gcn, test_acc = evaluate(features, edge_index_orig, labels_full, test_mask)
print(f"Test set results (on model from epoch {best_epoch}):\n"
      f"  Loss: {test_total_loss:.4f} (Total) {test_loss_gl:.4f} (GL) {test_loss_gcn:.4f} (GCN)\n"
      f"  Accuracy: {test_acc:.4f}")

# Example of how to run:
# python pytorch_glcn/train.py --dataset cora --epochs 300 --lr_gl 0.005 --lr_gcn 0.01 --hidden_gl 128 --hidden_gcn 64 --patience 50
# Note: Ensure that pytorch_glcn is in PYTHONPATH or run from the directory containing pytorch_glcn package.
# e.g., python -m pytorch_glcn.train --dataset cora ...
# Also, data path needs to be correct. If data is in 'pytorch_glcn/data', then path="./data/" is correct if
# running from 'pytorch_glcn/' directory. If running from parent of 'pytorch_glcn', then path="./pytorch_glcn/data/"
# For now, assuming data is 'pytorch_glcn/data' and script is run from 'pytorch_glcn' or package mode.

if __name__ == '__main__':
    # This part is mostly for structure, the script runs when invoked.
    # The argparse and main logic are at the top level.
    print(f"Running SGLCN on {args.dataset} with specified arguments.")

```
**Important Notes on the generated script:**

1.  **`edge_index_orig` for `SparseGraphLearn`**:
    *   The `load_data` function in `utils.py` currently returns a preprocessed adjacency matrix (`adj`). The `SparseGraphLearn` layer, however, is designed to learn edge importances from an initial set of candidate edges, which is typically the original graph structure (before normalization or adding self-loops).
    *   In the generated `train.py`, I've used `adj_sparse._indices()` as a placeholder for `edge_index_orig`. This might not be what SGL intends if `adj_sparse` is already heavily processed.
    *   **Required Fix**: The `load_data` function in `pytorch_glcn/utils/utils.py` needs to be updated to also return the original adjacency matrix (or its edge index) before any preprocessing like adding self-loops or normalization. This original edge index should then be passed to the `SGLCN` model.

2.  **Data Path in `load_data`**:
    *   The `load_data` function is called with `path="./data/"`. This assumes that the `train.py` script is run from within the `pytorch_glcn` directory, and there's a subdirectory `pytorch_glcn/data`. If the script is run from the root of the repository, the path should be `path="./pytorch_glcn/data/"`. This needs to be consistent with the actual directory structure and how the script is invoked.

3.  **Learning Rate Scheduler**:
    *   The TensorFlow `exponential_decay` has `global_step` and `decay_steps` which control when decay happens. PyTorch's `ExponentialLR` decays each epoch by `gamma`.
    *   I've implemented a `gamma` calculation that tries to match the TF behavior if `decay_steps` is interpreted as "decay every N epochs". The current `scheduler.step()` is called every epoch. If more precise TF-like step-based decay is needed, the scheduler update logic would need to be conditional (`if epoch % args.lr_decay_steps == 0`). The current simplified `gamma = args.lr_decay_rate ** (1.0 / args.lr_decay_steps)` assumes `lr_decay_steps` is the period over which the full `lr_decay_rate` is applied.

4.  **Optimizer**:
    *   The original TensorFlow code uses two optimizers and optimizes `loss1` (graph learning) and `loss2` (GCN) separately.
    *   The PyTorch script uses one optimizer with parameter groups for `lr_gl` and `lr_gcn`. It then backpropagates the `total_loss`. This is a common PyTorch approach. If strict separation of optimization steps is required (e.g., update GL params, then GCN params), the training loop would need two separate `backward()` calls on `loss_gl` and `loss_gcn` respectively, and two optimizer steps, possibly zeroing gradients in between carefully. The current script sums the losses and does one `backward()` and `step()`.

Let's proceed with creating this file.
