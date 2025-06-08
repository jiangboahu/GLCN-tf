import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SparseGraphLearn, GraphConvolution
# from .inits import glorot_, zeros_ # Example if using custom inits externally

class SGLCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_gl_dim, hidden_gcn_dim,
                 dropout_rate=0.5, weight_decay=5e-4, loss_lr1=0.1, loss_lr2=0.1,
                 sparse_inputs=True): # Add other FLAGS-like params
        super(SGLCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_gl_dim = hidden_gl_dim
        self.hidden_gcn_dim = hidden_gcn_dim
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.loss_lr1 = loss_lr1 # For graph learning feature smoothness
        self.loss_lr2 = loss_lr2 # For graph learning S sparsity/trace
        self.sparse_inputs = sparse_inputs

        # SparseGraphLearn layer
        self.gl_layer = SparseGraphLearn(input_dim=self.input_dim,
                                         output_dim=self.hidden_gl_dim,
                                         dropout=self.dropout_rate,
                                         sparse_inputs=self.sparse_inputs,
                                         act=F.relu)

        # GraphConvolution layers
        # Original GCN uses input features with learned graph S
        self.gcn_layer1 = GraphConvolution(input_dim=self.input_dim, # Original features
                                           output_dim=self.hidden_gcn_dim,
                                           dropout=self.dropout_rate,
                                           sparse_inputs=self.sparse_inputs, # Input features can be sparse
                                           act=F.relu)

        self.gcn_layer2 = GraphConvolution(input_dim=self.hidden_gcn_dim,
                                           output_dim=self.output_dim,
                                           dropout=self.dropout_rate,
                                           sparse_inputs=False, # Input to this layer is dense output of gcn_layer1
                                           act=lambda x: x) # No activation for logits

    def forward(self, features, edge_index_orig):
        """
        features: Input node features (sparse or dense)
        edge_index_orig: Original graph structure (LongTensor [2, num_edges]),
                         used by SparseGraphLearn to define potential edges.
        """
        # Graph learning layer
        # h_gl are the embeddings from the SparseGraphLearn layer
        # S_learned is the learned sparse adjacency matrix
        h_gl, S_learned = self.gl_layer(features, edge_index_orig)

        # GCN layers using the learned graph S_learned
        # The original paper/code uses the *original* features as input to the first GCN layer,
        # not the h_gl from the graph learning layer.
        x_gcn1 = self.gcn_layer1(features, S_learned)
        outputs = self.gcn_layer2(x_gcn1, S_learned)

        return outputs, h_gl, S_learned

    def l2_loss(self, layers):
        loss = 0.0
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    loss += torch.norm(param, p=2)**2
        return loss

    def loss_fn(self, outputs, h_gl, S_learned, labels, labels_mask, num_nodes):
        """
        Calculates the total loss for the SGLCN model.
        outputs: Logits from the model's forward pass.
        h_gl: Node embeddings from the SparseGraphLearn layer.
        S_learned: Learned sparse adjacency matrix (torch.sparse_coo_tensor).
        labels: Ground truth labels (one-hot or class indices).
        labels_mask: Boolean tensor indicating which samples to use for loss.
        num_nodes: Total number of nodes in the graph.
        """

        # --- Loss for Graph Learning Part (loss1 in original) ---
        loss_gl = 0.0

        # L2 regularization for SparseGraphLearn layer parameters
        loss_gl += self.weight_decay * self.l2_loss([self.gl_layer])

        # Graph Learning loss terms
        # Term 1: Feature smoothness over the learned graph (tf.trace(tf.matmul(tf.transpose(self.x), D) @ self.x))
        # D = I - S (as derived from original code: D_tf = -I; D_tf = sparse_add(D_tf,S_tf)*-1 = (S_tf-I)*-1 = I-S_tf)
        # So, D_torch = Identity - S_learned_dense
        # loss_trace_features = trace(h_gl.T @ (I - S_learned_dense) @ h_gl)

        # S_learned is sparse, convert to dense for operations if necessary and feasible
        # This can be memory-intensive for large graphs.
        # TODO: Explore if this can be computed efficiently with sparse tensors.
        if S_learned.is_sparse:
            S_learned_dense = S_learned.to_dense()
        else:
            S_learned_dense = S_learned # Should be sparse from SparseGraphLearn

        # Check if S_learned_dense is empty or num_nodes is zero
        if num_nodes > 0 and S_learned_dense.numel() > 0 :
            identity_matrix = torch.eye(num_nodes, device=h_gl.device)
            D_matrix = identity_matrix - S_learned_dense

            # Ensure h_gl has correct dimensions for matmul
            if h_gl.shape[0] == num_nodes:
                 # h_gl: [N, F_gl], D_matrix: [N, N]
                term1_val = torch.trace(h_gl.t() @ D_matrix @ h_gl)
                loss_gl += self.loss_lr1 * term1_val
            else:
                # This case should ideally not happen if inputs are correct
                print(f"Warning: h_gl shape {h_gl.shape} not compatible with num_nodes {num_nodes} for loss term 1.")


        # Term 2: Sparsity/Trace of S (tf.trace(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S))))
        # This is equivalent to trace(S_learned_dense.T @ S_learned_dense) or sum(S_learned_dense * S_learned_dense)
        # For sparse S, this is S_learned._values().pow(2).sum() if S is symmetric and S.T @ S.
        # Original code: tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S)
        # This implies S.T @ S_dense. If S is symmetric (often the case for GCNs), S.T = S.
        # So, trace(S_dense @ S_dense) or trace(S_dense.T @ S_dense)
        if S_learned_dense.numel() > 0: # Check if S_learned_dense is not empty
            # If S_learned is guaranteed to be symmetric, S.T @ S = S @ S
            # tf.sparse_tensor_dense_matmul(tf.sparse_transpose(S), tf.sparse_tensor_to_dense(S))
            # This looks like sum_{i,j} S_ji * S_ij. If S is symmetric, sum S_ij^2
            # Or, more generally, trace(S.T @ S)
            term2_val = torch.trace(S_learned_dense.t() @ S_learned_dense)
            loss_gl -= self.loss_lr2 * term2_val


        # --- Loss for GCN Part (loss2 in original) ---
        loss_gcn = 0.0

        # L2 regularization for GCN layers' parameters
        # Original code only adds L2 for layers1 (first GCN layer) to loss2
        loss_gcn += self.weight_decay * self.l2_loss([self.gcn_layer1, self.gcn_layer2])


        # Cross-entropy loss
        # Ensure labels are Long type for CrossEntropyLoss if they are class indices
        if labels.dim() > 1 and labels.shape[1] > 1: # One-hot encoded
            labels_target = torch.argmax(labels, dim=1)
        else: # Class indices
            labels_target = labels.long() # Ensure it's Long type

        # Compute cross-entropy loss for all samples
        ce_loss_full = F.cross_entropy(outputs, labels_target, reduction='none')

        # Apply mask
        if labels_mask.dtype != torch.bool:
            labels_mask = labels_mask.bool()

        masked_ce_loss = ce_loss_full[labels_mask]

        if masked_ce_loss.numel() > 0:
            loss_gcn += masked_ce_loss.mean()
        else:
            # Handle case where mask is all False, to avoid NaN from mean of empty tensor
            loss_gcn += 0.0

        total_loss = loss_gl + loss_gcn
        return total_loss, loss_gl, loss_gcn

    def accuracy(self, outputs, labels, labels_mask):
        """
        Calculates accuracy.
        outputs: Logits from the model.
        labels: Ground truth labels (one-hot or class indices).
        labels_mask: Boolean tensor.
        """
        if labels_mask.dtype != torch.bool:
            labels_mask = labels_mask.bool()

        if labels_mask.sum() == 0: # No samples to calculate accuracy for
            return torch.tensor(0.0, device=outputs.device)

        preds = torch.argmax(outputs, dim=1)

        if labels.dim() > 1 and labels.shape[1] > 1: # One-hot encoded
            labels_target = torch.argmax(labels, dim=1)
        else: # Class indices
            labels_target = labels

        correct = (preds == labels_target)[labels_mask]
        return correct.float().mean()

    def get_optimizer_param_groups(self, lr_gl, lr_gcn):
        """
        Returns parameter groups for different learning rates.
        One group for SparseGraphLearn layer, another for GCN layers.
        """
        params_gl = [
            {'params': self.gl_layer.parameters(), 'lr': lr_gl, 'weight_decay': 0} # L2 already in loss_gl
        ]
        params_gcn = [
            {'params': self.gcn_layer1.parameters(), 'lr': lr_gcn, 'weight_decay': 0}, # L2 already in loss_gcn
            {'params': self.gcn_layer2.parameters(), 'lr': lr_gcn, 'weight_decay': 0}  # L2 already in loss_gcn
        ]
        # If L2 is handled by optimizer's weight_decay, then set it here and remove from manual loss.
        # For this implementation, L2 is manually added to the loss terms.
        return params_gl, params_gcn

# Example usage (pseudo-code, actual training loop would be more detailed)
# model = SGLCN(input_dim, output_dim, hidden_gl, hidden_gcn)
# features, adj_orig_edge_index, labels, train_mask, num_nodes = load_some_data()
# optimizer_gl = torch.optim.Adam(model.get_optimizer_param_groups()[0])
# optimizer_gcn = torch.optim.Adam(model.get_optimizer_param_groups()[1])

# model.train()
# optimizer_gl.zero_grad()
# optimizer_gcn.zero_grad()

# logits, h_gl, S_learned = model(features, adj_orig_edge_index)
# total_loss, loss_gl_val, loss_gcn_val = model.loss_fn(logits, h_gl, S_learned, labels, train_mask, num_nodes)

# # The original paper uses a sequential optimization strategy:
# # 1. Optimize GL part (loss_gl)
# # 2. Optimize GCN part (loss_gcn)
# # This can be achieved by calling backward on individual losses and stepping respective optimizers,
# # or having a combined loss and one optimizer with parameter groups.
# # For now, let's assume we might optimize them separately if needed, or sum them.
# # If optimizing separately:
# # loss_gl_val.backward(retain_graph=True) # if loss_gcn_val needs h_gl gradients
# # optimizer_gl.step()
# # loss_gcn_val.backward()
# # optimizer_gcn.step()

# # Or if total_loss is used with an optimizer that handles param groups:
# # optimizer = torch.optim.Adam(model.get_optimizer_param_groups(lr1, lr2))
# # total_loss.backward()
# # optimizer.step()

# acc = model.accuracy(logits, labels, train_mask)
