import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np

class LandVisorTrainer:
    def __init__(
        self, 
        student_model, 
        dataloader_train, 
        dataloader_val, 
        criterion_pfce, 
        optimizer, 
        device,
        ema_alpha=0.99, 
        consistency_weight=0.1,
        num_classes=7
    ):
        self.device = device
        self.student = student_model.to(device)
        
        # Initialize Teacher as a copy of Student (no gradients)
        self.teacher = copy.deepcopy(student_model).to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.train_loader = dataloader_train
        self.val_loader = dataloader_val
        self.criterion_pfce = criterion_pfce
        self.optimizer = optimizer
        
        self.ema_alpha = ema_alpha
        self.consistency_weight = consistency_weight
        self.num_classes = num_classes

    def _update_teacher(self):
        """Exponential Moving Average update of teacher weights."""
        with torch.no_grad():
            for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
                t_param.data.mul_(self.ema_alpha).add_(s_param.data, alpha=1 - self.ema_alpha)

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.train() # Teacher stays in train mode for dropout/batchnorm consistency
        
        metrics = {"loss": 0, "loss_sup": 0, "loss_cons": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for images, sparse_masks, _ in pbar:
            images = images.to(self.device)
            sparse_masks = sparse_masks.to(self.device)

            # 1. Student Forward Pass
            s_logits = self.student(images)
            s_probs = F.softmax(s_logits, dim=1)

            # 2. Supervised Loss: pfCE on labeled points
            loss_sup = self.criterion_pfce(s_logits, sparse_masks)

            # 3. Teacher Forward Pass (Pseudo-labeling)
            with torch.no_grad():
                # Add slight noise to images for the teacher to enforce robustness
                t_images = images + torch.randn_like(images) * 0.02
                t_logits = self.teacher(t_images)
                t_probs = F.softmax(t_logits, dim=1)

            # 4. Consistency Loss: MSE on unlabeled pixels
            # We mask out pixels that already have labels to focus on the unknown
            unlabeled_mask = (sparse_masks == self.criterion_pfce.ignore_index).unsqueeze(1).float()
            loss_cons = F.mse_loss(s_probs * unlabeled_mask, t_probs * unlabeled_mask)

            # 5. Total Loss & Optimization
            total_loss = loss_sup + (self.consistency_weight * loss_cons)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 6. Smooth Teacher Update
            self._update_teacher()

            # Update logging
            metrics["loss"] += total_loss.item()
            metrics["loss_sup"] += loss_sup.item()
            metrics["loss_cons"] += loss_cons.item()
            pbar.set_postfix({"total": total_loss.item(), "sup": loss_sup.item()})

        return {k: v / len(self.train_loader) for k, v in metrics.items()}  
    @torch.no_grad()
    def validate(self):
        """
        Evaluates the Teacher model against the full ground truth masks.
        """
        self.teacher.eval()
        
        # Tracking metrics for the whole validation set
        total_iou = 0.0
        class_iou = {cls: [] for cls in range(self.num_classes)}
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for images, _, full_masks in pbar:
            images = images.to(self.device)
            full_masks = full_masks.to(self.device)

            # Use the Teacher for inference as it's the more stable model
            outputs = self.teacher(images)
            preds = torch.argmax(outputs, dim=1)

            # Calculate mIoU for this batch
            batch_iou, per_class_dict = self._calculate_batch_metrics(preds, full_masks)
            
            total_iou += batch_iou
            for cls, val in per_class_dict.items():
                class_iou[cls].append(val)

        # Average results
        avg_miou = total_iou / len(self.val_loader)
        final_class_iou = {cls: np.mean(vals) if vals else 0 for cls, vals in class_iou.items()}
        
        print(f"\nValidation mIoU: {avg_miou:.4f}")
        return avg_miou, final_class_iou

    def _calculate_batch_metrics(self, preds, targets):
        """Internal helper to compute IoU per batch."""
        ious = []
        per_class = {}
        
        for cls in range(self.num_classes):
            intersection = ((preds == cls) & (targets == cls)).sum().float()
            union = ((preds == cls) | (targets == cls)).sum().float()
            
            if union == 0:
                # If class isn't in target or prediction, we skip it to avoid NaN
                continue
            
            iou = (intersection / union).item()
            ious.append(iou)
            per_class[cls] = iou
            
        return (sum(ious) / len(ious)) if ious else 0.0, per_class
    