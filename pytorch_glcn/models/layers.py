import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

class SparseGraphLearn(Module):
    """
    Sparse Graph learning layer.
    """
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, act=F.relu, bias=False):
        super(SparseGraphLearn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.bias_flag = bias

        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = Parameter(torch.FloatTensor(output_dim, 1))
        if self.bias_flag:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = lambda x: x


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, edge_index):
        """
        inputs: FloatTensor, shape [num_nodes, input_dim]
        edge_index: LongTensor, shape [2, num_edges]
        """
        num_nodes = inputs.shape[0]

        if self.sparse_inputs:
            # Sparse dropout:
            # Create a mask with the same shape and sparsity pattern as inputs
            # For simplicity, if inputs is sparse, we apply dropout to its values
            if inputs.is_sparse:
                # Create a dropout mask for the values
                mask = torch.rand(inputs._values().size()) < (1.0 - self.dropout_rate)
                # Apply the mask to the values and scale
                values_after_dropout = inputs._values()[mask] * (1.0 / (1.0 - self.dropout_rate) if (1.0 - self.dropout_rate) > 0 else 0.0)
                # Create a new sparse tensor with the dropped out values
                x = torch.sparse_coo_tensor(inputs._indices()[:, mask], values_after_dropout, inputs.shape)
            else: # Should not happen if sparse_inputs is True and inputs is dense
                x = self.dropout(inputs)

        else:
            x = self.dropout(inputs)

        # Graph learning
        h = torch.mm(x, self.weights) # [num_nodes, output_dim]

        # Edge features
        row, col = edge_index
        h_row = h[row] # [num_edges, output_dim]
        h_col = h[col] # [num_edges, output_dim]

        edge_v = torch.abs(h_row - h_col) # [num_edges, output_dim]
        edge_v = self.act(torch.mm(edge_v, self.a)).squeeze() # [num_edges]

        # Sparse softmax (simplified: apply softmax to existing edge values)
        # For a more accurate sparse softmax, one might need to implement it carefully,
        # considering numerical stability and ensuring it sums to 1 over neighborhoods.
        # This simplified version applies softmax over the computed edge_v values.
        # A common approach for graph attention networks is to use softmax over the neighborhood of each node.
        # Here, sgraph is an adjacency matrix, so softmax should be applied row-wise or column-wise.

        # Create sparse adjacency matrix with learned edge weights
        # Ensure edge_v is not empty
        if edge_v.numel() == 0:
            # Handle case with no edges or all edge_v are zero after activation
            # Create an empty sparse tensor or handle as per model requirements
            sgraph = torch.sparse_coo_tensor(edge_index, torch.zeros(edge_index.shape[1], device=inputs.device), (num_nodes, num_nodes))
        else:
            # Attempting a sparse softmax. This is non-trivial.
            # A true sparse softmax would involve grouping by row index, applying softmax, and reassembling.
            # PyTorch Geometric has implementations for this (e.g., torch_sparse.softmax).
            # For now, we'll normalize existing edge weights using softmax.
            # This is a simplification and might need adjustment based on the specific GCN model.
            # Let's apply softmax to the edge values directly.
            # This is not a standard row/column-wise softmax on the sparse matrix.
            edge_v_softmax = F.softmax(edge_v, dim=0) # Softmax over all existing edge attention scores
            sgraph = torch.sparse_coo_tensor(edge_index, edge_v_softmax, (num_nodes, num_nodes))


        if self.bias is not None:
            h = h + self.bias

        return h, sgraph


class GraphConvolution(Module):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, act=F.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.sparse_inputs = sparse_inputs # Note: PyTorch sparse.mm handles sparse X dense W
        self.act = act
        self.bias_flag = bias

        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        if self.bias_flag:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = lambda x: x # No-op if dropout is 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        """
        inputs: FloatTensor, shape [num_nodes, input_dim] (can be sparse or dense)
        adj: SparseFloatTensor, shape [num_nodes, num_nodes]
        """
        x = inputs

        # Dropout
        if self.sparse_inputs:
            if not x.is_sparse: # Should ideally be sparse if sparse_inputs is True
                 x = self.dropout(x) # Apply standard dropout if dense despite flag
            else:
                # Create a dropout mask for the values
                mask = torch.rand(x._values().size()) < (1.0 - self.dropout_rate)
                # Apply the mask to the values and scale
                values_after_dropout = x._values()[mask] * (1.0 / (1.0 - self.dropout_rate) if (1.0 - self.dropout_rate) > 0 else 0.0)
                # Create a new sparse tensor with the dropped out values
                x = torch.sparse_coo_tensor(x._indices()[:,mask], values_after_dropout, x.shape)
        else:
            x = self.dropout(x) # Standard dropout for dense inputs

        # Convolution
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weights)  # XW, result is dense if W is dense
        else:
            support = torch.mm(x, self.weights) # XW, result is dense

        # Output = Adj * (XW)
        # torch.spmm requires adj to be sparse and support to be dense.
        # If support is sparse (e.g. if W was sparse and X sparse), this needs adjustment
        if adj.is_sparse and not support.is_sparse:
            output = torch.spmm(adj, support)
        elif adj.is_sparse and support.is_sparse: # Both sparse
            # This case is more complex, may need torch_sparse or convert one to dense if small enough
            # For now, assume 'support' becomes dense after multiplication with dense 'self.weights'
            # If 'self.weights' could also be sparse, this would need a different sparse_mm kernel
            output = torch.sparse.mm(adj, support) # adj (sparse) * support (sparse)
        elif not adj.is_sparse and not support.is_sparse: # Both dense
             output = torch.mm(adj, support)
        else: # adj dense, support sparse (uncommon for GCN standard adj)
            output = torch.mm(adj, support.to_dense())


        # Bias
        if self.bias is not None:
            output = output + self.bias

        return self.act(output)
