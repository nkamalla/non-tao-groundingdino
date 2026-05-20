import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from groundingdino.util.box_ops import generalized_box_iou


class GroundingDINOLoss(nn.Module):
    def __init__(
        self,
        loss_coef_bbox=5.0,
        loss_coef_giou=2.0,
    ):
        super().__init__()
        self.loss_coef_bbox = loss_coef_bbox
        self.loss_coef_giou = loss_coef_giou

    def box_cxcywh_to_xyxy(self, x):
        """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def giou_loss(self, outputs, targets):
        try:
            if outputs.shape[0] == 0 or targets.shape[0] == 0:
                return torch.tensor(0.0, device=outputs.device)
            giou = generalized_box_iou(outputs, targets)
            return (1 - torch.diag(giou)).mean()
        except Exception as e:
            return torch.tensor(0.0, device=outputs.device)

    def Hungarian_matching(self, outputs, targets):
        """Match predicted and GT boxes using GIoU"""
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]  # [B, 900, 4] in [x, y, w, h]

        indices = []

        for b in range(pred_boxes.shape[0]):
            if targets[b]["boxes"].shape[0] == 0:
                indices.append((np.array([]), np.array([])))
                continue

            gt_boxes = targets[b]["boxes"].to(device)  # [N, 4] in [x1, y1, x2, y2]
            
            # Convert pred boxes from [x, y, w, h] to [x1, y1, x2, y2]
            pred_boxes_b = self.box_cxcywh_to_xyxy(pred_boxes[b])  # [900, 4]
            pred_boxes_b = torch.clamp(pred_boxes_b, 0, 1)

            try:
                # Match based on GIoU
                giou_mat = generalized_box_iou(pred_boxes_b, gt_boxes)  # [900, N]
                cost = -giou_mat  # Negative for minimization

                row, col = linear_sum_assignment(cost.detach().cpu().numpy())
                indices.append((row, col))
            except Exception as e:
                indices.append((np.array([]), np.array([])))

        return indices

    def forward(self, outputs, targets, tokenized, tokenizer):
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]  # [B, 900, 4] in [x, y, w, h]

        B = pred_boxes.shape[0]

        indices = self.Hungarian_matching(outputs, targets)

        total_loss = 0
        bbox_loss_total = 0
        giou_loss_total = 0
        count = 0

        for b in range(B):
            row, col = indices[b]
            if len(row) == 0:
                continue

            # Convert pred boxes from [x, y, w, h] to [x1, y1, x2, y2]
            pred_boxes_b = self.box_cxcywh_to_xyxy(pred_boxes[b])  # [900, 4]
            pred_boxes_b = torch.clamp(pred_boxes_b, 0, 1)
            
            gt_boxes = targets[b]["boxes"].to(device)  # [N, 4] in [x1, y1, x2, y2]
            
            pred_box = pred_boxes_b[row]  # [num_matched, 4]
            gt_box = gt_boxes[col]  # [num_matched, 4]

            # Compute losses
            bbox_loss = F.l1_loss(pred_box, gt_box)
            giou_loss = self.giou_loss(pred_box, gt_box)
            
            loss = self.loss_coef_bbox * bbox_loss + self.loss_coef_giou * giou_loss

            total_loss += loss
            bbox_loss_total += bbox_loss.detach()
            giou_loss_total += giou_loss.detach()
            count += 1

        if count == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "bbox_loss": torch.tensor(0.0, device=device),
                "giou_loss": torch.tensor(0.0, device=device),
            }

        return {
            "loss": total_loss / count,
            "bbox_loss": bbox_loss_total / count,
            "giou_loss": giou_loss_total / count,
        }