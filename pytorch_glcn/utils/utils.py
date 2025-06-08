import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch

def sparse_to_torch_sparse_coo_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse COO tensor."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(coords, values, shape)

def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = 1
    return mask

def load_data(dataset_str, path="../data/"):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes_of_this_node]};
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting.

    All data is converted to torch Tensors.
    """
    data_path = path + dataset_str + "/"
    if dataset_str == "cora":
        features_mat = sio.loadmat(data_path + "feature.mat")
        features = sp.csr_matrix(features_mat['matrix'], dtype=np.float32)
        adj_mat = sio.loadmat(data_path + "adj.mat")
        adj = sp.csr_matrix(adj_mat['matrix'], dtype=np.float32)
        labels_mat = sio.loadmat(data_path + "label.mat")
        labels = torch.from_numpy(labels_mat['matrix']).long()

        idx_train = torch.arange(140)
        idx_val = torch.arange(200, 500)
        idx_test = torch.arange(500, 1500)

    elif dataset_str == "citeseer":
        features_mat = sio.loadmat(data_path + "feature.mat")
        features = sp.csr_matrix(features_mat['matrix'], dtype=np.float32)
        adj_mat = sio.loadmat(data_path + "adj.mat")
        adj = sp.csr_matrix(adj_mat['matrix'], dtype=np.float32)
        labels_mat = sio.loadmat(data_path + "label.mat")
        labels = torch.from_numpy(labels_mat['matrix']).long()

        idx_test_mat = sio.loadmat(data_path + "test.mat")
        idx_test = torch.from_numpy(idx_test_mat['array'].flatten()).long()
        idx_train = torch.arange(120)
        idx_val = torch.arange(120, 620)
    else:
        raise ValueError(f"Dataset {dataset_str} not supported.")

    # Symmetrize adjacency matrix (as in original GL-GCN utils)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Store original edge index before adding self-loops or normalization
    adj_coo = adj.tocoo()
    edge_index_orig = torch.from_numpy(
        np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64)
    )

    features = preprocess_features(features) # Row-normalize features
    adj_processed_for_gcn = preprocess_adj(adj) # Normalize adj with self-loops for GCN layers

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = torch.zeros_like(labels)
    y_val = torch.zeros_like(labels)
    y_test = torch.zeros_like(labels)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    # adj_processed_for_gcn is the one used by GCN layers.
    # edge_index_orig is for the SparseGraphLearn layer.
    return adj_processed_for_gcn, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, edge_index_orig

def preprocess_features(features):
    """Row-normalize feature matrix and convert to torch sparse COO tensor."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_torch_sparse_coo_tensor(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to torch sparse COO tensor."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_torch_sparse_coo_tensor(adj_normalized)
