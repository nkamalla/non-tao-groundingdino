import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss
from groundingdino.util.box_ops import generalized_box_iou


class GroundingDINOLoss(nn.Module):
    """
    Loss function for GroundingDINO fine-tuning on COCO-style datasets (e.g. URPC 2020).

    Key design principles:
    - pred_logits shape: [B, num_queries, max_text_len] — each query predicts
      a score per text token, NOT per fixed class index.
    - Classification targets must use positive maps: only the token positions
      corresponding to a GT object's class name should be activated.
    - Hungarian matching uses bbox L1 + GIoU + class-aware token costs.
    - GT boxes are expected in normalized [x1, y1, x2, y2] format.
    """

    def __init__(
        self,
        loss_coef_bbox=5.0,
        loss_coef_giou=2.0,
        loss_coef_cls=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.loss_coef_bbox = loss_coef_bbox
        self.loss_coef_giou = loss_coef_giou
        self.loss_coef_cls = loss_coef_cls
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def box_cxcywh_to_xyxy(self, x):
        """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
            dim=-1,
        )

    def giou_loss(self, pred_boxes, gt_boxes):
        """Compute mean (1 - GIoU) for paired boxes. Both in xyxy format."""
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        giou = generalized_box_iou(pred_boxes, gt_boxes)
        return (1 - torch.diag(giou)).mean()

    def build_positive_maps(self, targets, tokenized, tokenizer):
        """
        Build per-sample positive maps linking each GT box to token positions
        in the caption via its class label.

        Returns two lists (one per batch element each):
            binary_maps: [num_gt, max_text_len] with 1.0 at positive token positions
                         (used as focal loss targets)
            norm_maps:   [num_gt, max_text_len] normalized so each row sums to 1
                         (used for matching cost computation)
        """
        B = len(targets)
        max_text_len = tokenized["input_ids"].shape[1]
        binary_maps = []
        norm_maps = []

        for b in range(B):
            labels = targets[b].get("labels", [])
            caption = targets[b].get("caption", "")
            num_gt = len(labels)

            if num_gt == 0:
                empty = torch.zeros((0, max_text_len), device=tokenized["input_ids"].device)
                binary_maps.append(empty)
                norm_maps.append(empty)
                continue

            pos_map = torch.zeros(
                (num_gt, max_text_len),
                dtype=torch.float,
                device=tokenized["input_ids"].device,
            )

            caption_lower = caption.lower()

            for j, label in enumerate(labels):
                label_lower = label.lower().strip()
                # Find the label's character span in the caption
                start_char = caption_lower.find(label_lower)
                if start_char == -1:
                    # Try individual words if multi-word label not found as whole
                    for word in label_lower.split():
                        ws = caption_lower.find(word)
                        if ws == -1:
                            continue
                        we = ws + len(word)
                        beg_tok = tokenized.char_to_token(b, ws)
                        end_tok = tokenized.char_to_token(b, we - 1)
                        if beg_tok is not None and end_tok is not None:
                            pos_map[j, beg_tok : end_tok + 1] = 1.0
                    continue

                end_char = start_char + len(label_lower)
                # Map character positions to token positions
                beg_tok = tokenized.char_to_token(b, start_char)
                end_tok = tokenized.char_to_token(b, end_char - 1)

                # Fallback if boundary chars map to None
                if beg_tok is None:
                    beg_tok = tokenized.char_to_token(b, start_char + 1)
                if end_tok is None:
                    end_tok = tokenized.char_to_token(b, end_char - 2)

                if beg_tok is not None and end_tok is not None:
                    pos_map[j, beg_tok : end_tok + 1] = 1.0

            # Binary map: 1.0 at positive positions (for focal loss targets)
            binary_maps.append(pos_map.clone())

            # Normalized map: each row sums to 1 (for matching cost weighting)
            row_sums = pos_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_maps.append(pos_map / row_sums)

        return binary_maps, norm_maps

    def Hungarian_matching(self, outputs, targets, norm_maps):
        """
        Bipartite matching using combined bbox + GIoU + class-aware costs.
        Uses normalized positive maps for the classification cost so that
        multi-token words don't dominate single-token words.
        """
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]

        indices = []

        for b in range(pred_boxes.shape[0]):
            num_gt = targets[b]["boxes"].shape[0]
            if num_gt == 0:
                indices.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                continue

            gt_boxes = targets[b]["boxes"].to(device).float()
            gt_boxes = torch.clamp(gt_boxes, 0, 1)

            pred_boxes_b = self.box_cxcywh_to_xyxy(pred_boxes[b])
            pred_boxes_b = torch.clamp(pred_boxes_b, 0, 1)

            # BBox L1 cost: [num_queries, num_gt]
            bbox_cost = torch.cdist(pred_boxes_b, gt_boxes, p=1)

            # GIoU cost: [num_queries, num_gt]
            giou_cost = -generalized_box_iou(pred_boxes_b, gt_boxes)

            # Class-aware cost using NORMALIZED positive maps
            # This ensures fair comparison across classes with different token counts
            pred_probs = torch.sigmoid(pred_logits[b])  # [num_queries, max_text_len]
            pos_map_b = norm_maps[b]  # [num_gt, max_text_len] (normalized)

            # matmul: [num_queries, T] @ [T, num_gt] -> [num_queries, num_gt]
            cls_cost = -(pred_probs @ pos_map_b.t())

            # Combined cost
            cost = (
                self.loss_coef_bbox * bbox_cost
                + self.loss_coef_giou * giou_cost
                + self.loss_coef_cls * cls_cost
            )

            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((row.astype(np.int64), col.astype(np.int64)))

        return indices

    def forward(self, outputs, targets, tokenized=None, tokenizer=None):
        """
        Compute GroundingDINO losses.

        Args:
            outputs: dict with 'pred_logits' [B, Q, T] and 'pred_boxes' [B, Q, 4] (cxcywh)
            targets: list of dicts with 'boxes' [N, 4] (xyxy normalized),
                     'labels' (list of str), 'caption' (str)
            tokenized: tokenizer output for the batch captions
            tokenizer: the tokenizer instance (unused but kept for API compat)
        """
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]
        B = pred_boxes.shape[0]

        # Build positive maps:
        #   binary_maps: 1.0 at positive positions (for focal loss targets)
        #   norm_maps: normalized per-row (for matching cost)
        binary_maps, norm_maps = self.build_positive_maps(targets, tokenized, tokenizer)

        # Hungarian matching uses normalized maps for fair class-cost comparison
        indices = self.Hungarian_matching(outputs, targets, norm_maps)

        # --- Classification loss (token-level focal loss) ---
        # CRITICAL: use BINARY targets (1.0 at positive token positions).
        # Using normalized soft targets (e.g. 0.25) causes NaN because:
        # 1) BCE optimal point becomes p=0.25 instead of p=1.0, conflicting with matching
        # 2) Focal modulation (1-p_t)^gamma amplifies loss at wrong operating point
        # 3) The resulting gradient instability causes parameter explosion → NaN
        target_logits = torch.zeros_like(pred_logits)  # [B, Q, T]

        num_positive_tokens = 0  # Track total positive elements for normalization
        for b in range(B):
            row, col = indices[b]
            if len(row) == 0:
                continue
            # Assign BINARY positive map as target (1.0 at class token positions)
            target_logits[b, row] = binary_maps[b][col]
            num_positive_tokens += binary_maps[b][col].sum().item()

        # Normalize by number of positive TOKEN positions (not just num_boxes).
        # This prevents the loss from scaling with text length.
        # E.g., 5 objects with avg 3 tokens each = 15 positive elements.
        num_pos = max(num_positive_tokens, 1.0)

        cls_loss = sigmoid_focal_loss(
            pred_logits,
            target_logits,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_pos

        # --- BBox + GIoU losses ---
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_giou_loss = torch.tensor(0.0, device=device)
        matched_count = 0

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

            total_bbox_loss = total_bbox_loss + F.l1_loss(pred_box, gt_box, reduction="sum")
            total_giou_loss = total_giou_loss + self.giou_loss(pred_box, gt_box) * len(row)
            matched_count += len(row)

        if matched_count == 0:
            return {
                "loss": self.loss_coef_cls * cls_loss,
                "bbox_loss": torch.tensor(0.0, device=device),
                "giou_loss": torch.tensor(0.0, device=device),
                "cls_loss": cls_loss.detach(),
            }

        # Normalize box losses by number of matched pairs
        bbox_loss = total_bbox_loss / (matched_count * 4)  # 4 coords per box
        giou_loss = total_giou_loss / matched_count

        # Final combined loss
        final_loss = (
            self.loss_coef_bbox * bbox_loss
            + self.loss_coef_giou * giou_loss
            + self.loss_coef_cls * cls_loss
        )

        # Auxiliary losses (if decoder intermediate outputs are provided)
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"]:
                aux_loss_dict = self.forward(aux, targets, tokenized, tokenizer)
                final_loss = final_loss + 0.5 * aux_loss_dict["loss"]

        return {
            "loss": final_loss,
            "bbox_loss": bbox_loss.detach(),
            "giou_loss": giou_loss.detach(),
            "cls_loss": cls_loss.detach(),
        }