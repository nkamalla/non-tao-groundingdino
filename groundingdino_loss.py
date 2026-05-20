import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss
from groundingdino.util.box_ops import generalized_box_iou


class GroundingDINOLoss(nn.Module):
    def __init__(
        self,
        loss_coef_bbox=5.0,
        loss_coef_giou=2.0,
        loss_coef_cls=1.0,
    ):
        super().__init__()
        self.loss_coef_bbox = loss_coef_bbox
        self.loss_coef_giou = loss_coef_giou
        self.loss_coef_cls = loss_coef_cls

    # ✅ Convert [cx, cy, w, h] → [x1, y1, x2, y2]
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [
                x_c - 0.5 * w,
                y_c - 0.5 * h,
                x_c + 0.5 * w,
                y_c + 0.5 * h,
            ],
            dim=-1,
        )

    # ✅ GIoU loss
    def giou_loss(self, outputs, targets):
        if outputs.shape[0] == 0 or targets.shape[0] == 0:
            return torch.tensor(0.0, device=outputs.device)

        giou = generalized_box_iou(outputs, targets)
        return (1 - torch.diag(giou)).mean()

    # ✅ Hungarian Matching
    def Hungarian_matching(self, outputs, targets):
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]

        indices = []

        for b in range(pred_boxes.shape[0]):
            if targets[b]["boxes"].shape[0] == 0:
                indices.append((np.array([]), np.array([])))
                continue

            # ✅ Prepare GT
            gt_boxes = targets[b]["boxes"].to(device).float()
            gt_boxes = torch.clamp(gt_boxes, 0, 1)

            # ✅ Convert predictions
            pred_boxes_b = self.box_cxcywh_to_xyxy(pred_boxes[b])
            pred_boxes_b = torch.clamp(pred_boxes_b, 0, 1)

            pred_logits_b = pred_logits[b]

            try:
                # ✅ BBox L1 cost
                bbox_cost = torch.cdist(pred_boxes_b, gt_boxes, p=1)

                # ✅ GIoU cost
                giou_cost = -generalized_box_iou(pred_boxes_b, gt_boxes)

                # ✅ Improved classification cost
                cls_cost = -torch.sigmoid(pred_logits_b).max(-1)[0].unsqueeze(1)

                # ✅ Final cost
                cost = (
                    self.loss_coef_bbox * bbox_cost
                    + self.loss_coef_giou * giou_cost
                    + self.loss_coef_cls * cls_cost
                )

                row, col = linear_sum_assignment(cost.detach().cpu().numpy())
                indices.append((row, col))

            except Exception as e:
                print("🔥 Matching error:", e)
                raise e   # ❗ do NOT silently ignore

        return indices

    # ✅ Forward
    def forward(self, outputs, targets, tokenized=None, tokenizer=None):
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]

        B = pred_boxes.shape[0]

        # ✅ Matching
        indices = self.Hungarian_matching(outputs, targets)

        total_loss = 0
        bbox_loss_total = 0
        giou_loss_total = 0
        count = 0

        # ✅ Build classification targets (ONLY matched queries = positive)
        target_logits = torch.zeros_like(pred_logits)

        for b in range(B):
            row, col = indices[b]
            if len(row) == 0:
                continue
            target_logits[b, row] = 1.0

        # ✅ Classification loss
        cls_loss = sigmoid_focal_loss(
            pred_logits,
            target_logits,
            reduction="mean"
        )

        # ✅ BBox + GIoU losses
        for b in range(B):
            row, col = indices[b]
            if len(row) == 0:
                continue

            pred_boxes_b = self.box_cxcywh_to_xyxy(pred_boxes[b])
            pred_boxes_b = torch.clamp(pred_boxes_b, 0, 1)

            gt_boxes = targets[b]["boxes"].to(device).float()
            gt_boxes = torch.clamp(gt_boxes, 0, 1)

            pred_box = pred_boxes_b[row]
            gt_box = gt_boxes[col]

            bbox_loss = F.l1_loss(pred_box, gt_box)
            giou_loss = self.giou_loss(pred_box, gt_box)

            loss = (
                self.loss_coef_bbox * bbox_loss
                + self.loss_coef_giou * giou_loss
            )

            total_loss += loss
            bbox_loss_total += bbox_loss.detach()
            giou_loss_total += giou_loss.detach()
            count += 1

        # ✅ If no matches → fallback to cls loss
        if count == 0:
            return {
                "loss": cls_loss,
                "bbox_loss": torch.tensor(0.0, device=device),
                "giou_loss": torch.tensor(0.0, device=device),
                "cls_loss": cls_loss.detach(),
            }

        total_loss = total_loss / count

        # ✅ Final loss
        final_loss = total_loss + self.loss_coef_cls * cls_loss

        # ✅ Auxiliary losses (DETR-style)
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"]:
                aux_loss = self.forward(aux, targets)
                final_loss += 0.5 * aux_loss["loss"]

        return {
            "loss": final_loss,
            "bbox_loss": bbox_loss_total / count,
            "giou_loss": giou_loss_total / count,
            "cls_loss": cls_loss.detach(),
        }