import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialFocalCELoss(nn.Module):
    """
    Implements Partial Focal Cross Entropy (pfCE) for weakly supervised segmentation.
    
    Formula: 
    pfCE = Σ(Focal_loss(pre, GT) * MASK_labeled) / ΣMASK_labeled
    
    This loss only computes gradients for pixels that have been explicitly 
    annotated (points), ignoring the vast majority of unlabeled pixels.
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super(PartialFocalCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): Predictions from the model [Batch, Classes, H, W]
            target (torch.Tensor): Sparse ground truth labels [Batch, H, W]
                                   Unlabeled pixels should be set to self.ignore_index.
        """
        # 1. Create the binary mask for labeled points
        # MASK_labeled is 1 where we have a label, 0 where we don't.
        mask_labeled = (target != self.ignore_index).float()
        
        # 2. Calculate standard Cross Entropy loss per pixel (no reduction)
        # We must use reduction='none' to keep the spatial dimensions for the mask multiplication.
        # We still pass ignore_index to F.cross_entropy to ensure it zero-outs unlabeled indices internally.
        ce_loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction='none')
        
        # 3. Calculate the Focal component
        # pt is the probability of the correct class
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply optional class balancing alpha if provided
        if self.alpha is not None:
            # This assumes alpha is a tensor of weights for each class
            at = self.alpha.gather(0, target.flatten(1)).view_as(target)
            focal_weight = at * focal_weight

        # 4. Compute the numerator: Σ(Focal_loss * MASK_labeled)
        # Note: ce_loss already has 0s for ignore_index pixels, but we multiply by 
        # mask_labeled to be explicit and match the requested formula.
        numerator = (focal_weight * ce_loss * mask_labeled).sum()
        
        # 5. Compute the denominator: ΣMASK_labeled
        denominator = mask_labeled.sum()
        
        # --- EDGE CASE HANDLING ---
        # If an image or batch has ZERO labeled points, denominator will be 0.
        # Dividing by zero results in NaN, which crashes training.
        if denominator == 0:
            # Return a zero loss that still allows the gradient graph to exist
            return logits.sum() * 0.0
            
        # 6. Final pfCE calculation
        return numerator / denominator
