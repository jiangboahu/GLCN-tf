import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sparse_dropout(x, rate, training=True):
    """
    Dropout for sparse tensors.
    If training is False or rate is 0, returns the input tensor as is.
    If x is not sparse, applies F.dropout.
    Otherwise, performs dropout on the values of the sparse tensor.
    """
    if not training or rate == 0.0:
        return x

    if not x.is_sparse:
        return F.dropout(x, p=rate, training=training) # Use training flag

    if not x.is_coalesced():
        x = x.coalesce()

    values = x.values()

    # Create a random mask for values.
    # Keep probability is (1 - rate).
    value_keep_mask = (torch.rand(values.shape[0], device=x.device) >= rate)

    new_values = values[value_keep_mask]
    new_indices = x.indices()[:, value_keep_mask]

    if rate < 1.0 and rate > 0.0: # Avoid division by zero or no scaling if rate is 0
        new_values = new_values / (1.0 - rate)

    if new_indices.shape[1] == 0:
        return torch.sparse_coo_tensor(x.indices()[:,:0], [], x.shape, device=x.device)

    return torch.sparse_coo_tensor(new_indices, new_values, x.shape, device=x.device)


class SparseGraphLearn(nn.Module):
    def __init__(self, input_dim, output_dim, edge_indices, num_nodes, dropout=0., act=F.relu, bias=False, sparse_inputs=False): # bias here refers to bias for 'h' if any
        super(SparseGraphLearn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer('edge_indices', torch.as_tensor(edge_indices, dtype=torch.long))
        self.num_nodes = num_nodes
        self.dropout_rate = dropout
        self.act = act if act is not None else nn.Identity()
        # self.bias_flag = bias # Original TF layer for graph learning part did not have bias on 'h'
        self.sparse_inputs = sparse_inputs

        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim)) # For transforming X to H
        self.a = nn.Parameter(torch.FloatTensor(output_dim, 1)) # For scoring edges

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x_inputs):
        x_processed = x_inputs
        if self.dropout_rate > 0:
            if self.sparse_inputs:
                if not x_inputs.is_sparse:
                    x_processed = F.dropout(x_inputs, self.dropout_rate, training=self.training)
                else:
                    x_processed = sparse_dropout(x_inputs, self.dropout_rate, self.training)
            else:
                if x_inputs.is_sparse:
                    x_processed = F.dropout(x_inputs.to_dense(), self.dropout_rate, training=self.training)
                else:
                    x_processed = F.dropout(x_inputs, self.dropout_rate, training=self.training)

        if x_processed.is_sparse:
            # Ensure weights is compatible with sparse.mm if x_processed is sparse (which it is)
            h = torch.sparse.mm(x_processed, self.weights)
        else:
            h = torch.matmul(x_processed, self.weights)

        current_edge_indices = self.edge_indices.to(h.device)

        if current_edge_indices.shape[1] == 0: # No edges to learn from
            # Return h and an empty sparse graph
            sgraph_softmax = torch.sparse_coo_tensor(current_edge_indices, [], (self.num_nodes, self.num_nodes), device=h.device)
            return h, sgraph_softmax

        h_src = h.index_select(0, current_edge_indices[0])
        h_dst = h.index_select(0, current_edge_indices[1])

        edge_v_features = torch.abs(h_src - h_dst)
        edge_v_scores = torch.matmul(edge_v_features, self.a)
        edge_v_scores = self.act(edge_v_scores).squeeze(-1)

        sgraph = torch.sparse_coo_tensor(current_edge_indices, edge_v_scores, (self.num_nodes, self.num_nodes), device=h.device)
        sgraph = sgraph.coalesce() # Important for correctness of sparse_softmax

        try:
            # dim=1 for row-wise softmax.
            sgraph_softmax = torch.sparse.softmax(sgraph, dim=1)
        except Exception as e: # Catch a broader exception for safety for now
            # This can happen if rows are all zero, leading to 0/0 in softmax for those rows.
            # tf.sparse_softmax might handle this by keeping zero rows as zero.
            # A robust solution might involve manual softmax or using a library like torch_sparse.
            # For now, if PyTorch's native sparse_softmax fails, we log and return unnormalized graph.
            print(f"Warning: torch.sparse.softmax failed (sgraph has {sgraph._nnz()} non-zero, shape {sgraph.shape}). Error: {e}. Returning un-softmaxed graph.")
            sgraph_softmax = sgraph
            # A possible strategy for empty rows:
            # Check sgraph.to_dense().sum(dim=1) == 0 for rows that are all zero.
            # tf.sparse_softmax produces a result where rows that were originally all zero will remain all zero.
            # PyTorch's torch.sparse.softmax might error or produce NaN if a row sum is zero.
            # A workaround for torch.sparse.softmax:
            # Convert to dense, add epsilon to zero rows, softmax, convert back to sparse. (Complex, changes graph)
            # Or, use a custom implementation that handles zero-sum rows.
            # For this conversion, we'll rely on torch.sparse.softmax and note this potential issue.

        return h, sgraph_softmax


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu, bias=False, sparse_inputs=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.act = act if act is not None else nn.Identity()
        self.bias_flag = bias
        self.sparse_inputs = sparse_inputs

        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if self.bias_flag:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_inputs, adj):
        x_processed = x_inputs
        if self.dropout_rate > 0:
            if self.sparse_inputs:
                if not x_inputs.is_sparse:
                    x_processed = F.dropout(x_inputs, self.dropout_rate, training=self.training)
                else:
                    x_processed = sparse_dropout(x_inputs, self.dropout_rate, self.training)
            else:
                if x_inputs.is_sparse:
                    x_processed = F.dropout(x_inputs.to_dense(), self.dropout_rate, training=self.training)
                else:
                    x_processed = F.dropout(x_inputs, self.dropout_rate, training=self.training)

        if x_processed.is_sparse:
            pre_sup = torch.sparse.mm(x_processed, self.weights)
        else:
            pre_sup = torch.matmul(x_processed, self.weights)

        if not adj.is_sparse: # Ensure adj is sparse
            adj = adj.to_sparse() # This is a fallback, adj should be prepared as sparse.

        output = torch.sparse.mm(adj, pre_sup)

        if self.bias is not None:
            output = output + self.bias

        return self.act(output)
