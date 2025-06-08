import torch
import torch.nn as nn
import torch.nn.functional as F
# Ensure torch_layers is importable. If glcn is a package, 'from .torch_layers import ...'
# If running scripts directly from glcn/, 'from torch_layers import ...' might work.
# For robustness with potential package structure:
from .torch_layers import SparseGraphLearn, GraphConvolution

class SGLCN_PyTorch(nn.Module):
    def __init__(self, input_dim, output_dim_labels, edge_indices, num_nodes,
                 hidden_gl_dim, hidden_gcn_dim, dropout_rate,
                 sparse_features=True): # sparse_features refers to the nature of the input 'features'
        super(SGLCN_PyTorch, self).__init__()

        self.input_dim = input_dim
        self.output_dim_labels = output_dim_labels # e.g., number of classes
        self.num_nodes = num_nodes

        # Layer definitions
        # 1. SparseGraphLearn layer
        # TF: self.layers0 = SparseGraphLearn(input_dim=self.input_dim,
        #                                 output_dim=FLAGS.hidden_gl,
        #                                 edge=self.edge, # From placeholders
        #                                 placeholders=self.placeholders, # For dropout, num_nodes etc.
        #                                 act=tf.nn.relu,
        #                                 dropout=True, # Translates to passing dropout_rate
        #                                 sparse_inputs=True) # This was True in TF
        self.layer0_sg_learn = SparseGraphLearn(input_dim=input_dim,
                                              output_dim=hidden_gl_dim,
                                              edge_indices=edge_indices,
                                              num_nodes=num_nodes,
                                              dropout=dropout_rate,
                                              act=F.relu,
                                              sparse_inputs=sparse_features)

        # 2. First GraphConvolution layer
        # TF: self.layers1 = GraphConvolution(input_dim=self.input_dim, # Original input_dim
        #                                     output_dim=FLAGS.hidden_gcn,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     sparse_inputs=True,
        #                                     logging=self.logging)
        # Note: input to this layer in TF was original self.inputs (features), not self.x from layer0
        self.layer1_gcn = GraphConvolution(input_dim=input_dim, # Takes original features
                                           output_dim=hidden_gcn_dim,
                                           dropout=dropout_rate,
                                           act=F.relu,
                                           bias=True, # GCN layers usually have bias, TF version had it.
                                           sparse_inputs=sparse_features)

        # 3. Second GraphConvolution layer (output layer)
        # TF: self.layers2 = GraphConvolution(input_dim=FLAGS.hidden_gcn,
        #                                     output_dim=self.output_dim, # Num classes
        #                                     placeholders=self.placeholders,
        #                                     act=lambda x: x, # Identity activation
        #                                     dropout=True,
        #                                     logging=self.logging)
        self.layer2_gcn_out = GraphConvolution(input_dim=hidden_gcn_dim,
                                               output_dim=output_dim_labels,
                                               dropout=dropout_rate,
                                               act=None, # Will default to nn.Identity in GraphConvolution
                                               bias=True, # GCN layers usually have bias
                                               sparse_inputs=False) # Input to this layer (x1) is dense

    def forward(self, features_input, adj_learned_S_override=None):
        # features_input: Original node features (sparse or dense based on self.layer0_sg_learn.sparse_inputs)
        # adj_learned_S_override: Optional override for the learned graph S, mainly for specific loss calculations or analysis.

        # 1. Graph Learning Layer
        # TF: self.x, self.S = self.layers0(self.inputs)
        # self.x was the transformed features from SparseGraphLearn (h in torch_layers)
        # self.S was the learned sparse graph (sgraph_softmax in torch_layers)

        # If adj_learned_S_override is provided, layer0 might not be needed to produce S,
        # but it's still needed for 'x' (transformed features for one part of the loss).
        # However, the typical forward pass uses the S learned by layer0.

        learned_x, learned_S = self.layer0_sg_learn(features_input)

        current_S = adj_learned_S_override if adj_learned_S_override is not None else learned_S

        # 2. First GCN Layer
        # TF: x1 = self.layers1(self.inputs, self.S)
        # It uses the *original* features_input, but the *learned* graph current_S.
        x1_gcn = self.layer1_gcn(features_input, current_S)

        # 3. Second GCN Layer (Output Layer)
        # TF: self.outputs = self.layers2(x1, self.S)
        # It uses the output of the first GCN (x1_gcn) and the *learned* graph current_S.
        # The input x1_gcn from a GCN layer is typically dense.
        # So, sparse_inputs for layer2_gcn_out should be False. (Corrected in __init__)
        final_outputs = self.layer2_gcn_out(x1_gcn, current_S)

        # The model returns all components that might be needed for the complex loss function
        # learned_x: features transformed by SparseGraphLearn (used in one part of TF loss)
        # learned_S: the learned graph structure (used in both parts of TF loss)
        # final_outputs: the final classification scores (used for cross-entropy part of TF loss)
        return final_outputs, learned_x, learned_S

    def get_weights_for_l2_loss(self):
        """
        Helper to get parameters for L2 regularization, mimicking TF's var grouping.
        vars1 = tf.trainable_variables()[0:2] # weights and 'a' from SparseGraphLearn
        vars2 = tf.trainable_variables()[2:]  # weights (and biases) from GCN layers

        This grouping is a bit fragile if layer structure changes.
        A more robust way is to access named parameters.
        """
        # Parameters from SparseGraphLearn (layer0)
        # self.layer0_sg_learn.weights, self.layer0_sg_learn.a
        params_loss1 = [self.layer0_sg_learn.weights, self.layer0_sg_learn.a]

        # Parameters from GCN layers (layer1 and layer2_gcn_out)
        params_loss2 = []
        params_loss2.extend([p for p in self.layer1_gcn.parameters()])
        params_loss2.extend([p for p in self.layer2_gcn_out.parameters()])

        return params_loss1, params_loss2

# Example usage (for testing structure, not for actual run):
# if __name__ == '__main__':
#     # Mock data
#     num_nodes = 100
#     num_features = 50
#     num_classes = 7
#     edge_indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long) # Sample edges
#
#     model = SGLCN_PyTorch(input_dim=num_features,
#                           output_dim_labels=num_classes,
#                           edge_indices=edge_indices,
#                           num_nodes=num_nodes,
#                           hidden_gl_dim=30,
#                           hidden_gcn_dim=20,
#                           dropout_rate=0.5,
#                           sparse_features=True) # Assuming input features are sparse
#
#     # Create dummy sparse features
#     # Typically, features would be prepared by a utility function
#     feat_indices = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long)
#     feat_values = torch.ones(feat_indices.shape[1])
#     sparse_features_input = torch.sparse_coo_tensor(feat_indices, feat_values, (num_nodes, num_features))
#
#     # Test forward pass
#     model.train() # Set to training mode for dropout
#     final_outputs, learned_x, learned_S = model(sparse_features_input)
#
#     print("Final Outputs Shape:", final_outputs.shape)
#     print("Learned X Shape:", learned_x.shape)
#     print("Learned S is sparse:", learned_S.is_sparse)
#     if learned_S.is_sparse:
#         print("Learned S non-zero elements:", learned_S._nnz())
