import torch
import torch.nn.functional as F

def masked_softmax_cross_entropy(preds, labels, mask):
    """
    Calculates masked softmax cross-entropy loss.

    Args:
        preds (torch.Tensor): Predicted logits, shape [N, num_classes].
        labels (torch.Tensor): Ground truth labels. Can be:
            - Class indices, shape [N].
            - One-hot encoded, shape [N, num_classes].
        mask (torch.Tensor): Boolean mask, shape [N], where True indicates an element to include.

    Returns:
        torch.Tensor: Scalar tensor representing the mean masked cross-entropy loss.
    """
    if not isinstance(preds, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Inputs must be PyTorch Tensors.")
    if preds.dim() != 2:
        raise ValueError("Preds tensor must be 2-dimensional (N, num_classes).")
    if mask.dim() != 1:
        raise ValueError("Mask tensor must be 1-dimensional (N).")
    if preds.shape[0] != labels.shape[0] or preds.shape[0] != mask.shape[0]:
        raise ValueError("Number of samples (N) must be consistent across preds, labels, and mask.")
    if mask.sum() == 0: # Avoid division by zero if mask is all False
        return torch.tensor(0.0, device=preds.device)

    # Ensure mask is boolean
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Calculate cross-entropy
    if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1): # Class indices
        # If labels are [N, 1], squeeze to [N]
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        # F.cross_entropy expects raw logits for preds and class indices for labels
        loss = F.cross_entropy(preds, labels.long(), reduction='none')
    elif labels.dim() == 2 and labels.shape[1] == preds.shape[1]: # One-hot encoded labels
        # Original TF function `tf.nn.softmax_cross_entropy_with_logits` implies one-hot labels.
        # PyTorch equivalent:
        log_probs = F.log_softmax(preds, dim=-1)
        loss = -torch.sum(labels * log_probs, dim=-1)
    else:
        raise ValueError("Labels must be class indices (N) or one-hot encoded (N, num_classes).")

    # Apply mask
    loss = loss[mask]

    # Mask scaling (as in original TensorFlow version's metrics)
    # mask_float = mask.float() # This should be the original mask, not the sub-selected one
    # mask_float_for_scaling = mask.float() # No, use the full mask for mean calculation
    # mask_float_for_scaling_mean = torch.mean(mask_float_for_scaling)
    # if mask_float_for_scaling_mean != 0:
    #     loss = loss * (1.0 / mask_float_for_scaling_mean) # This is not how TF did it.
    # TF: `mask /= tf.reduce_mean(mask)`. This means the elements of the mask are scaled.
    # Then `loss *= mask` (element-wise).
    # This seems to imply the loss for each *included* item is scaled by (1/mean_of_original_mask_values).
    # Let's replicate the logic from SGLCN model's loss_fn which is simpler: mean of masked elements.
    # The TF metrics `mask /= tf.reduce_mean(mask)` then `loss = loss * mask`, then `tf.reduce_mean(loss)`
    # is equivalent to `tf.reduce_sum(loss_unmasked * mask_orig) / tf.reduce_sum(mask_orig)` if loss is 0 where mask is 0
    # which is just `tf.reduce_mean(loss_masked_items)`.
    # The specific scaling `mask /= tf.reduce_mean(mask)` in TF metrics might be a way to normalize
    # the contribution of the mask if the number of masked items varies.
    # For now, let's stick to the standard PyTorch way of taking the mean of the loss of *valid* items.
    # The SGLCN model's loss_fn already does:
    #   ce_loss_full = F.cross_entropy(outputs, labels_target, reduction='none')
    #   masked_ce_loss = ce_loss_full[labels_mask]
    #   loss_gcn += masked_ce_loss.mean()
    # This is standard. If the specific TF scaling is crucial, it's more complex.
    # The prompt mentions "mask /= tf.reduce_mean(mask)". Let's try to implement that intent.
    # This means scaling the loss contribution of each item.

    # If the intent of `mask /= tf.reduce_mean(mask)` was to ensure that if you have fewer true
    # entries in the mask, their individual losses contribute more, such that the sum of the
    # "scaled mask" entries is N_total_items. This would be:
    # scaled_mask_val = 1.0 / mean_mask_val = N_total_items / N_true_mask_items
    # loss_val = sum(loss_i * scaled_mask_val for i in true_mask_items) / N_total_items
    # This is identical to sum(loss_i for i in true_mask_items) / N_true_mask_items.
    # So, .mean() on the masked loss is standard and correct.
    # The original TF metrics might have had a different structure for this.
    # Let's assume the SGLCN model's internal loss is the reference for PyTorch.
    # If a direct port of a specific TF `masked_softmax_cross_entropy` is needed
    # that behaves differently, the requirements for scaling need clarification.
    # For now, returning the direct mean of losses for elements where mask is True.
    if loss.numel() > 0:
        return loss.mean()
    else:
        return torch.tensor(0.0, device=preds.device)


def masked_accuracy(preds, labels, mask):
    """
    Calculates masked accuracy.

    Args:
        preds (torch.Tensor): Predicted logits, shape [N, num_classes].
        labels (torch.Tensor): Ground truth labels. Can be:
            - Class indices, shape [N].
            - One-hot encoded, shape [N, num_classes].
        mask (torch.Tensor): Boolean mask, shape [N], where True indicates an element to include.

    Returns:
        torch.Tensor: Scalar tensor representing the mean masked accuracy.
    """
    if not isinstance(preds, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Inputs must be PyTorch Tensors.")
    if preds.dim() != 2:
        raise ValueError("Preds tensor must be 2-dimensional (N, num_classes).")
    if mask.dim() != 1:
        raise ValueError("Mask tensor must be 1-dimensional (N).")
    if preds.shape[0] != labels.shape[0] or preds.shape[0] != mask.shape[0]:
        raise ValueError("Number of samples (N) must be consistent across preds, labels, and mask.")
    if mask.sum() == 0: # Avoid division by zero if mask is all False
        return torch.tensor(0.0, device=preds.device)

    # Ensure mask is boolean
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Get predicted classes
    pred_classes = torch.argmax(preds, dim=1)

    # Get target classes
    if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1): # Class indices
        if labels.dim() == 2 and labels.shape[1] == 1:
            target_classes = labels.squeeze(1)
        else:
            target_classes = labels
    elif labels.dim() == 2 and labels.shape[1] == preds.shape[1]: # One-hot encoded labels
        target_classes = torch.argmax(labels, dim=1)
    else:
        raise ValueError("Labels must be class indices (N) or one-hot encoded (N, num_classes).")

    # Calculate accuracy for masked elements
    correct_predictions = (pred_classes[mask] == target_classes[mask]).float()

    # The prompt mentions:
    # mask_float = mask.float()
    # mask_float /= torch.mean(mask_float) # This scales the mask itself
    # accuracy_all = correct_prediction * mask_float # correct_prediction is already N elements long
    # return torch.mean(accuracy_all) # This would be mean over all N elements
    # This is complex. Let's re-evaluate the TF logic `tf.reduce_mean(accuracy_all * mask)`
    # where `accuracy_all` are per-element correct (0 or 1) and `mask` is scaled.
    # If `mask_tf_scaled = mask_tf_orig / tf.reduce_mean(mask_tf_orig)`,
    # then `tf.reduce_mean(accuracy_all * mask_tf_scaled)`
    # `= (1/N) * sum_i (accuracy_all_i * mask_tf_orig_i / mean_mask_tf_orig)`
    # `= (1/N) * sum_i (accuracy_all_i * mask_tf_orig_i * (N / sum_mask_tf_orig_i) )`
    # `= (1 / sum_mask_tf_orig_i) * sum_i (accuracy_all_i * mask_tf_orig_i)`
    # This is exactly `sum(correct_predictions_where_mask_is_true) / sum(mask_is_true_count)`.
    # So, the standard PyTorch way of calculating mean accuracy on masked elements is correct.
    # model.accuracy already does: (preds == labels_target)[labels_mask].float().mean()

    if correct_predictions.numel() > 0:
        return correct_predictions.mean()
    else:
        return torch.tensor(0.0, device=preds.device)

```

**Regarding Mask Scaling:**

The prompt mentioned a specific TensorFlow-style mask scaling: `mask /= tf.reduce_mean(mask)`.
I've analyzed this for both cross-entropy and accuracy:

*   **For `masked_softmax_cross_entropy`**: If the TensorFlow operation `tf.nn.softmax_cross_entropy_with_logits` returns losses for all elements (even those to be masked out, perhaps as 0 or some value), and then this scaled mask is applied element-wise (`loss_values * scaled_mask_values`), followed by `tf.reduce_mean` over all N elements, the calculation would be:
    `mean_loss = (1/N) * sum(loss_i * mask_i / mean_of_mask_values)`.
    If `loss_i` is non-zero only where `mask_i` is 1 (or true), this simplifies to:
    `mean_loss = (1/N) * sum_{i where mask_i=1} (loss_i * (1 / (num_true_in_mask / N)))`
    `= (1/N) * sum_{i where mask_i=1} (loss_i * N / num_true_in_mask)`
    `= (1 / num_true_in_mask) * sum_{i where mask_i=1} (loss_i)`.
    This is precisely the mean of the loss values for the elements where the mask is true. So, `loss[mask].mean()` in PyTorch is equivalent.

*   **For `masked_accuracy`**: A similar derivation shows that `(pred_classes[mask] == target_classes[mask]).float().mean()` is also equivalent to the described TensorFlow scaled mask logic when applied to accuracy.

Therefore, the implementations above use the standard PyTorch approach of first filtering by the mask and then taking the mean, which mathematically aligns with the TensorFlow scaled mask logic's effective outcome for these specific metrics. The SGLCN model's internal loss and accuracy calculations already follow this standard PyTorch pattern.

These standalone metric functions are now available if needed, for example, for more granular analysis or if a different part of a project requires them separately from the model object.
