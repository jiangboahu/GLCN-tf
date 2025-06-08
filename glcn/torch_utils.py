import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch

def scipy_sparse_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape)

def sample_mask(idx, l):
    """Create mask as boolean Torch tensor."""
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = True
    return mask

def load_data(dataset_str="cora", path="../data/"):
    """
    Loads input data from MATLAB files for the specified dataset,
    and returns PyTorch tensors.
    Adjacency matrix and features are returned as SciPy sparse matrices initially,
    to be processed further by preprocessing functions.
    Masks and labels are returned as PyTorch tensors.
    """
    data_path = path + dataset_str + "/"

    if dataset_str == "cora":
        features_mat = sio.loadmat(data_path + "feature.mat")
        features = features_mat['matrix'] # Scipy sparse matrix
        adj_mat = sio.loadmat(data_path + "adj.mat")
        adj = adj_mat['matrix'] # Scipy sparse matrix
        labels_mat = sio.loadmat(data_path + "label.mat")
        labels = torch.from_numpy(labels_mat['matrix'].astype(np.float32)) # One-hot labels

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset_str == "citeseer":
        features_mat = sio.loadmat(data_path + "feature.mat")
        features = features_mat['matrix']
        adj_mat = sio.loadmat(data_path + "adj.mat")
        adj = adj_mat['matrix']
        labels_mat = sio.loadmat(data_path + "label.mat")
        labels = torch.from_numpy(labels_mat['matrix'].astype(np.float32))

        idx_test_mat = sio.loadmat(data_path + "test.mat")
        idx_test = idx_test_mat['array'].flatten() # In Citeseer, test indices are explicitly given
        idx_train = range(120)
        idx_val = range(120, 620) # Ensure this range is correct as per original data split
    else:
        raise ValueError(f"Dataset {dataset_str} not supported.")

    # Symmetrize adjacency matrix (if not already symmetric)
    # This operation ensures that if (i,j) is an edge, (j,i) is also an edge.
    # It takes the union of edges from A and A.T.
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T < adj)

    num_nodes = labels.shape[0]
    train_mask = sample_mask(idx_train, num_nodes)
    val_mask = sample_mask(idx_val, num_nodes)
    test_mask = sample_mask(idx_test, num_nodes)

    y_train = torch.zeros_like(labels)
    y_val = torch.zeros_like(labels)
    y_test = torch.zeros_like(labels)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features_sp_matrix):
    """
    Row-normalize feature matrix and convert to a sparse PyTorch tensor.
    Input: features_sp_matrix (Scipy sparse matrix)
    Output: PyTorch sparse tensor
    """
    rowsum = np.array(features_sp_matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features_normalized = r_mat_inv.dot(features_sp_matrix)
    return scipy_sparse_to_torch_sparse_tensor(features_normalized)

def normalize_adj_torch(adj_sp_matrix):
    """
    Symmetrically normalize adjacency matrix (D^-0.5 * A * D^-0.5)
    The input adj_sp_matrix here is assumed to be the original adjacency matrix.
    Self-loops are added within this function.
    Input: adj_sp_matrix (Scipy sparse matrix)
    Output: PyTorch sparse tensor
    """
    adj_with_self_loops = adj_sp_matrix.tocoo() + sp.eye(adj_sp_matrix.shape[0], format='coo')
    adj_with_self_loops.eliminate_zeros() # Remove explicit zeros if any resulted from addition
    # Ensure unique edges by summing data for duplicate (row, col) pairs, then keep unique.
    adj_with_self_loops = adj_with_self_loops.tocsr().tocoo()

    rowsum = np.array(adj_with_self_loops.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # D^-0.5 * (A+I) * D^-0.5
    adj_normalized = adj_with_self_loops.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return scipy_sparse_to_torch_sparse_tensor(adj_normalized)


def preprocess_graph_data(adj_original_sp, features_original_sp):
    """
    Processes the original adjacency and feature matrices.
    - Normalizes features.
    - Normalizes adjacency matrix for GCN layers (A_norm_gcn).
    - Extracts edge indices for the graph learning layer (edge_indices_gl) from A+I structure.

    Returns:
        processed_features: PyTorch sparse tensor (normalized features)
        adj_norm_gcn: PyTorch sparse tensor (normalized adj for GCN layers)
        edge_indices_gl: PyTorch LongTensor [2, num_edges] (for SparseGraphLearn layer)
    """
    processed_features = preprocess_features(features_original_sp) # Returns torch sparse tensor

    # For GCN layers: D^-0.5 * (A_orig+I) * D^-0.5
    # normalize_adj_torch adds self-loops internally before normalization
    adj_norm_gcn = normalize_adj_torch(adj_original_sp.copy()) # Returns torch sparse tensor

    # For SparseGraphLearn layer's edge_indices:
    # These edges define the initial set of possible connections for graph learning.
    # Derived from the structure of (A_original + I).
    adj_plus_self_loops = adj_original_sp.tocoo() + sp.eye(adj_original_sp.shape[0], format='coo')
    adj_plus_self_loops.eliminate_zeros() # Clean up any zero entries

    # Ensure unique edges: convert to CSR (sums duplicates) then back to COO.
    # This ensures that (row, col) pairs are unique, which is expected for edge_indices.
    adj_plus_self_loops = adj_plus_self_loops.tocsr().tocoo()

    edge_indices_gl = torch.from_numpy(
        np.vstack((adj_plus_self_loops.row, adj_plus_self_loops.col)).astype(np.int64)
    )

    return processed_features, adj_norm_gcn, edge_indices_gl
