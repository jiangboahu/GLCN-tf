import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# Assuming torch_utils.py and torch_models.py are in the same directory (glcn/)
# or glcn is in PYTHONPATH.
from .torch_utils import load_data, preprocess_graph_data
from .torch_models import SGLCN_PyTorch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset string: "cora" or "citeseer".')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr1', type=float, default=0.005, help='Initial learning rate for graph learning layer params.')
    parser.add_argument('--lr2', type=float, default=0.005, help='Initial learning rate for GCN layer params.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight for L2 loss.')
    parser.add_argument('--early_stopping', type=int, default=100, help='Tolerance for early stopping (# of epochs).')
    parser.add_argument('--hidden_gl', type=int, default=70, help='Number of units in GraphLearning hidden layer.')
    parser.add_argument('--hidden_gcn', type=int, default=30, help='Number of units in GCN hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
    parser.add_argument('--losslr1', type=float, default=0.01, help='Coefficient for graph smoothness term in loss1 (X^T(I-S)X).')
    parser.add_argument('--losslr2', type=float, default=0.0001, help='Coefficient for S Frobenius norm term in loss1 (-||S||_F^2).')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    return parser.parse_args()

def masked_softmax_cross_entropy(preds, labels, mask):
    log_softmax_preds = F.log_softmax(preds, dim=1)
    loss = -torch.sum(log_softmax_preds * labels, dim=1) # Element-wise cross entropy
    mask = mask.float()
    loss = loss * mask
    # Add epsilon for safety to prevent division by zero if mask sum is 0
    return torch.sum(loss) / (torch.sum(mask) + 1e-8)

def masked_accuracy(preds, labels, mask):
    mask_bool = mask.bool()
    preds_classes = torch.argmax(preds, dim=1)
    true_classes = torch.argmax(labels, dim=1) # Assuming labels are one-hot

    if torch.sum(mask_bool).item() == 0: # Handle case where mask is empty
        return torch.tensor(0.0, device=preds.device)

    correct_prediction = (preds_classes[mask_bool] == true_classes[mask_bool])
    # Add epsilon for safety if mask_bool sum is 0 (already handled by check above)
    return torch.sum(correct_prediction).float() / torch.sum(mask_bool).float()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    adj_orig, features_orig, labels_full, _, _, _, train_mask, val_mask, test_mask = load_data(args.dataset, path="./data/") # Adjusted path
    features, adj_norm_gcn, edge_indices_gl = preprocess_graph_data(adj_orig, features_orig)

    features = features.to(device)
    # adj_norm_gcn is used by GCN layers within the model if needed, but SGLCN_PyTorch uses learned_S
    # For SGLCN, the adj_norm_gcn is not directly passed to the model instance during forward pass,
    # as the model learns its own graph S. However, some GCN models might take it.
    # Let's ensure it's on the device if any part of the model (even if not SGLCN) might expect it.
    adj_norm_gcn = adj_norm_gcn.to(device)
    edge_indices_gl = edge_indices_gl.to(device)
    labels_full = labels_full.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    num_nodes = labels_full.shape[0]
    input_dim = features.shape[1]
    output_dim_labels = labels_full.shape[1]

    model = SGLCN_PyTorch(input_dim=input_dim,
                          output_dim_labels=output_dim_labels,
                          edge_indices=edge_indices_gl,
                          num_nodes=num_nodes,
                          hidden_gl_dim=args.hidden_gl,
                          hidden_gcn_dim=args.hidden_gcn,
                          dropout_rate=args.dropout,
                          sparse_features=features.is_sparse).to(device)

    params_loss1_group, params_loss2_group = model.get_weights_for_l2_loss()

    optimizer1 = optim.Adam(params_loss1_group, lr=args.lr1)
    optimizer2 = optim.Adam(params_loss2_group, lr=args.lr2)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    test_acc_at_best_val = 0.0

    print(f"Starting training on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t_epoch_start = time.time()
        model.train()

        final_outputs, learned_x, learned_S = model(features)

        loss1 = 0.0
        if args.weight_decay > 0:
            # L2 regularization for parameters in the first group (SparseGraphLearn)
            for param in params_loss1_group:
                loss1 += args.weight_decay * torch.sum(param * param) / 2

        # Graph smoothness term: tr(H^T * (I-S) * H)
        # H is learned_x. (I-S)H = learned_x - S @ learned_x
        term1_intermediate = learned_x - torch.sparse.mm(learned_S, learned_x)
        # Element-wise product sum: sum(diag(H^T * (I-S) * H))
        loss1_term1 = args.losslr1 * torch.sum(learned_x * term1_intermediate)
        loss1 += loss1_term1

        # Frobenius norm term for S: - ||S||_F^2
        if learned_S._nnz() > 0 :
             loss1_term2 = -args.losslr2 * torch.sum(learned_S.values() * learned_S.values())
             loss1 += loss1_term2

        # Classification loss
        loss2 = masked_softmax_cross_entropy(final_outputs, labels_full, train_mask)
        if args.weight_decay > 0:
            # L2 regularization for parameters in the second group (GCN layers)
            for param in params_loss2_group:
                loss2 += args.weight_decay * torch.sum(param * param) / 2

        optimizer1.zero_grad()
        if not isinstance(loss1, torch.Tensor): # Ensure loss1 is a tensor
             # If loss1 became a float (e.g. if weight_decay is 0 and both terms are 0 due to empty S)
             # It needs to be a tensor for backward()
            loss1 = torch.tensor(loss1, device=device, requires_grad=True if model.training else False)
        # Zero gradients for both optimizers before backward passes
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Ensure loss1 is a tensor (it might be a float if no L2 and S is empty)
        if not isinstance(loss1, torch.Tensor):
            loss1 = torch.tensor(loss1, device=device, requires_grad=True if model.training else False)

        # Calculate gradients for both losses with respect to their dependent parameters
        # loss1 gradients are for params_loss1_group
        # loss2 gradients are for params_loss2_group AND params_loss1_group (via learned_S)
        loss1.backward(retain_graph=True) # retain_graph because learned_S computation is needed by loss2.backward()
        loss2.backward()

        # Apply updates
        optimizer1.step()
        optimizer2.step()

        model.eval()
        with torch.no_grad():
            eval_outputs, _, _ = model(features) # Use the original features for evaluation

            val_loss = masked_softmax_cross_entropy(eval_outputs, labels_full, val_mask).item()
            val_acc = masked_accuracy(eval_outputs, labels_full, val_mask).item()

            test_acc = masked_accuracy(eval_outputs, labels_full, test_mask).item()

        loss1_item = loss1.item() if isinstance(loss1, torch.Tensor) else loss1
        loss2_item = loss2.item()

        print(f"Epoch: {epoch+1:04d} "
              f"loss1={loss1_item:.5f} loss2={loss2_item:.5f} "
              f"val_loss={val_loss:.5f} val_acc={val_acc:.5f} test_acc={test_acc:.5f} "
              f"time={time.time() - t_epoch_start:.5f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc_at_best_val = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Optionally save checkpoint here
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping and args.early_stopping > 0: # Ensure early_stopping is positive
            print(f"Early stopping at epoch {epoch+1} due to no improvement in val_loss for {args.early_stopping} epochs.")
            break

    print("Optimization Finished!")
    print(f"Best Epoch: {best_epoch:04d}, Best Val Loss: {best_val_loss:.5f}, Test Acc at Best Val: {test_acc_at_best_val:.5f}")

if __name__ == '__main__':
    args = parse_args()
    # Update data path to be relative to where torch_train.py is likely executed from (e.g. repo root)
    # The load_data function has default path "../data/", so if script is in glcn/, ../data/ is fine.
    # If script is run from repo root as `python glcn/torch_train.py`, then data path needs to be "data/"
    # The run scripts (torch_run_cora.py) are in glcn/ and call main(args).
    # If load_data's default path is `../data/`, and run scripts are in `glcn/`, then `../data/` becomes `glcn/../data/` relative to CWD if CWD is `glcn/`.
    # If CWD is repo root, then `glcn/../data/` refers to `data/`.
    # For consistency, let's ensure load_data is called with a path that works assuming CWD is repo root.
    # The run scripts call main(args), so args are passed. We can modify load_data call in main.
    # Path in load_data default is `../data/`.
    # If `torch_run_cora.py` is in `glcn/` and run from `glcn/` (e.g. `cd glcn; python torch_run_cora.py`), `../data/` is correct.
    # If run from repo root (`python glcn/torch_run_cora.py`), `../data/` inside `load_data` (which is in `glcn/torch_utils.py`) would be relative to `glcn/`, so `glcn/../data/` which is `data/`. This is also correct.
    # The provided code for `load_data` in `torch_utils.py` has `path="../data/"`.
    # This path is relative to where `torch_utils.py` *thinks* it is or where the CWD is.
    # To be safe, let's make it explicit in the main function of torch_train.
    # The line `adj_orig, features_orig, labels_full, _, _, _, train_mask, val_mask, test_mask = load_data(args.dataset, path="./data/")`
    # assumes that the script is run from the repository root.
    # If scripts are run from glcn/ (e.g. `cd glcn; python torch_run_cora.py`), then path should be `../data/`.
    # Let's stick to the original `load_data` default path logic and assume execution from repo root for `python glcn/script.py`
    # The `path` argument in `load_data` is `../data/`. If `torch_utils.py` is in `glcn/`, then `../data/` relative to `glcn/` is `data/` at the repo root.
    # So, if `main()` is called from `glcn/torch_run_cora.py` (which is in `glcn/`), and `load_data` is in `glcn/torch_utils.py`,
    # its default `path="../data/"` will correctly point to `glcn/../data/` = `data/` relative to the repo root. This seems fine.
    # No change needed for `load_data` call path argument if `torch_utils.py`'s `load_data` default is `../data/`.

    print("Running with args:", args)
    main(args)
