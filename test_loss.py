"""
Unit test for GroundingDINOLoss to verify correctness with COCO-style URPC 2020 data.
Tests:
1. Positive map construction (token-to-class mapping)
2. Hungarian matching with class-aware costs
3. Gradient flow through loss
4. Edge cases (empty GT, single GT, multiple classes)
"""
import torch
from transformers import AutoTokenizer
from groundingdino_loss import GroundingDINOLoss


def make_tokenized(tokenizer, captions, device="cpu"):
    """Tokenize captions matching train.py's approach."""
    return tokenizer(
        captions,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)


def test_positive_map_construction():
    """Test that positive maps correctly link class labels to token positions."""
    print("=" * 60)
    print("TEST 1: Positive Map Construction")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loss_fn = GroundingDINOLoss()

    # Simulate URPC 2020 with 4 categories
    caption = "holothurian echinus scallop starfish ."
    targets = [{
        "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        "labels": ["holothurian", "scallop"],
        "caption": caption,
    }]

    tokenized = make_tokenized(tokenizer, [caption])

    positive_maps = loss_fn.build_positive_maps(targets, tokenized, tokenizer)
    pos_map = positive_maps[0]

    print(f"Caption: '{caption}'")
    print(f"Labels: {targets[0]['labels']}")
    print(f"Positive map shape: {pos_map.shape}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])}")

    # Verify each row sums to ~1
    row_sums = pos_map.sum(dim=-1)
    print(f"Row sums (should be ~1.0): {row_sums}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Row sums should be 1.0, got {row_sums}"

    # Verify 'holothurian' maps to correct tokens (NOT 'scallop' tokens)
    holo_tokens = pos_map[0].nonzero(as_tuple=True)[0]
    scallop_tokens = pos_map[1].nonzero(as_tuple=True)[0]
    print(f"'holothurian' token positions: {holo_tokens.tolist()}")
    print(f"'scallop' token positions: {scallop_tokens.tolist()}")

    # They should not overlap
    overlap = set(holo_tokens.tolist()) & set(scallop_tokens.tolist())
    assert len(overlap) == 0, f"Token positions should not overlap, got: {overlap}"

    print("✅ PASSED: Positive maps correctly constructed\n")


def test_loss_gradient_flow():
    """Test that loss produces valid gradients."""
    print("=" * 60)
    print("TEST 2: Loss Gradient Flow")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loss_fn = GroundingDINOLoss()

    B, Q, T = 2, 900, 20  # batch=2, queries=900, text_len depends on tokenization
    caption1 = "holothurian echinus scallop starfish ."
    caption2 = "echinus starfish ."

    tokenized = make_tokenized(tokenizer, [caption1, caption2])
    T = tokenized["input_ids"].shape[1]

    # Simulated model outputs (requires grad)
    pred_logits = torch.randn(B, Q, T, requires_grad=True)
    pred_boxes_raw = torch.randn(B, Q, 4, requires_grad=True)
    pred_boxes = pred_boxes_raw.sigmoid()  # Non-leaf (like model output)
    pred_boxes.retain_grad()

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    targets = [
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.8, 0.9]]),
            "labels": ["holothurian", "scallop"],
            "caption": caption1,
        },
        {
            "boxes": torch.tensor([[0.2, 0.3, 0.6, 0.7]]),
            "labels": ["starfish"],
            "caption": caption2,
        },
    ]

    loss_dict = loss_fn(outputs, targets, tokenized, tokenizer)

    print(f"Loss: {loss_dict['loss'].item():.6f}")
    print(f"BBox loss: {loss_dict['bbox_loss'].item():.6f}")
    print(f"GIoU loss: {loss_dict['giou_loss'].item():.6f}")
    print(f"Cls loss: {loss_dict['cls_loss'].item():.6f}")

    # Check gradient flows
    loss_dict["loss"].backward()
    assert pred_logits.grad is not None, "No gradient on pred_logits!"
    assert pred_boxes_raw.grad is not None, "No gradient on pred_boxes!"
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN!"
    assert not torch.isinf(loss_dict["loss"]), "Loss is inf!"

    print(f"pred_logits grad norm: {pred_logits.grad.norm().item():.6f}")
    print(f"pred_boxes grad norm: {pred_boxes_raw.grad.norm().item():.6f}")
    print("✅ PASSED: Gradients flow correctly\n")


def test_empty_targets():
    """Test with no GT boxes in a batch element."""
    print("=" * 60)
    print("TEST 3: Empty Targets Handling")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loss_fn = GroundingDINOLoss()

    caption1 = "holothurian echinus ."
    caption2 = "scallop starfish ."

    tokenized = make_tokenized(tokenizer, [caption1, caption2])
    T = tokenized["input_ids"].shape[1]

    B, Q = 2, 900
    pred_logits = torch.randn(B, Q, T, requires_grad=True)
    pred_boxes = torch.sigmoid(torch.randn(B, Q, 4, requires_grad=True))

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    # First sample has GT, second has NONE
    targets = [
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.4, 0.5]]),
            "labels": ["holothurian"],
            "caption": caption1,
        },
        {
            "boxes": torch.zeros((0, 4)),
            "labels": [],
            "caption": caption2,
        },
    ]

    loss_dict = loss_fn(outputs, targets, tokenized, tokenizer)
    loss_dict["loss"].backward()

    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN with empty targets!"
    print(f"Loss with one empty target: {loss_dict['loss'].item():.6f}")
    print("✅ PASSED: Empty targets handled gracefully\n")


def test_all_empty_targets():
    """Test with ALL targets empty (no GT in entire batch)."""
    print("=" * 60)
    print("TEST 4: All Empty Targets")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loss_fn = GroundingDINOLoss()

    caption1 = "holothurian ."
    caption2 = "scallop ."

    tokenized = make_tokenized(tokenizer, [caption1, caption2])
    T = tokenized["input_ids"].shape[1]

    B, Q = 2, 900
    pred_logits = torch.randn(B, Q, T, requires_grad=True)
    pred_boxes = torch.sigmoid(torch.randn(B, Q, 4, requires_grad=True))

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    targets = [
        {"boxes": torch.zeros((0, 4)), "labels": [], "caption": caption1},
        {"boxes": torch.zeros((0, 4)), "labels": [], "caption": caption2},
    ]

    loss_dict = loss_fn(outputs, targets, tokenized, tokenizer)
    loss_dict["loss"].backward()

    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN with all empty!"
    print(f"Loss with all empty: {loss_dict['loss'].item():.6f}")
    print(f"bbox_loss: {loss_dict['bbox_loss'].item():.6f} (should be 0)")
    print(f"giou_loss: {loss_dict['giou_loss'].item():.6f} (should be 0)")
    print("✅ PASSED: All-empty batch handled\n")


def test_hungarian_matching_quality():
    """Test that matching assigns predictions close to GT boxes correctly."""
    print("=" * 60)
    print("TEST 5: Hungarian Matching Quality")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loss_fn = GroundingDINOLoss()

    caption = "holothurian scallop ."
    tokenized = make_tokenized(tokenizer, [caption])
    T = tokenized["input_ids"].shape[1]

    Q = 900
    # Place 2 predicted boxes very close to 2 GT boxes
    pred_boxes = torch.rand(1, Q, 4) * 0.1  # Random small boxes
    # Make query 5 close to GT[0] and query 10 close to GT[1]
    gt_boxes = torch.tensor([[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]])
    # In cxcywh format for predictions
    pred_boxes[0, 5] = torch.tensor([0.3, 0.3, 0.2, 0.2])  # center=(0.3,0.3), w=0.2, h=0.2
    pred_boxes[0, 10] = torch.tensor([0.7, 0.7, 0.2, 0.2])  # center=(0.7,0.7), w=0.2, h=0.2

    # Also set corresponding logits high for correct class tokens
    pred_logits = torch.zeros(1, Q, T) - 5.0  # All low
    pos_maps = loss_fn.build_positive_maps(
        [{"boxes": gt_boxes, "labels": ["holothurian", "scallop"], "caption": caption}],
        tokenized, tokenizer
    )
    # Make query 5 have high logits for 'holothurian' tokens
    pred_logits[0, 5] = 5.0 * pos_maps[0][0]
    # Make query 10 have high logits for 'scallop' tokens
    pred_logits[0, 10] = 5.0 * pos_maps[0][1]

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{
        "boxes": gt_boxes,
        "labels": ["holothurian", "scallop"],
        "caption": caption,
    }]

    indices = loss_fn.Hungarian_matching(outputs, targets, pos_maps)
    row, col = indices[0]

    print(f"Matched pred indices: {row}")
    print(f"Matched GT indices: {col}")

    # Query 5 should match GT 0 (holothurian), Query 10 should match GT 1 (scallop)
    assert 5 in row and 10 in row, f"Expected queries 5 and 10 to be matched, got {row}"
    idx5 = list(row).index(5)
    idx10 = list(row).index(10)
    assert col[idx5] == 0, f"Query 5 should match GT 0, got GT {col[idx5]}"
    assert col[idx10] == 1, f"Query 10 should match GT 1, got GT {col[idx10]}"

    print("✅ PASSED: Hungarian matching assigns correct pairs\n")


def test_coco_bbox_format():
    """Test COCO [x, y, w, h] -> [x1, y1, x2, y2] conversion in Dataset."""
    print("=" * 60)
    print("TEST 6: COCO BBox Format Conversion")
    print("=" * 60)

    # Simulate COCO annotation: [x, y, w, h] = [100, 150, 200, 300]
    # Image size: 640x480
    # Expected normalized xyxy: [100/640, 150/480, 300/640, 450/480]
    #                         = [0.15625, 0.3125, 0.46875, 0.9375]
    import json, tempfile, os
    from train import Dataset

    annotation = {
        "file_name": "dummy.jpg",
        "caption": "holothurian .",
        "annotations": [
            {"bbox": [100, 150, 200, 300], "category_name": "holothurian"}
        ]
    }

    # Create temporary JSONL and dummy image
    tmpdir = tempfile.mkdtemp()
    jsonl_path = os.path.join(tmpdir, "train.jsonl")
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(annotation) + "\n")

    # Create dummy image 640x480
    from PIL import Image
    img = Image.new("RGB", (640, 480))
    img.save(os.path.join(tmpdir, "dummy.jpg"))

    ds = Dataset(jsonl_path, tmpdir, bbox_format="xywh")
    _, target, _ = ds[0]

    expected = torch.tensor([[100/640, 150/480, 300/640, 450/480]])
    print(f"COCO bbox [x,y,w,h]: [100, 150, 200, 300]")
    print(f"Image size: 640x480")
    print(f"Expected normalized xyxy: {expected[0].tolist()}")
    print(f"Got: {target['boxes'][0].tolist()}")

    assert torch.allclose(target["boxes"], expected, atol=1e-5), \
        f"Box conversion failed! Expected {expected}, got {target['boxes']}"

    # Cleanup
    os.remove(jsonl_path)
    os.remove(os.path.join(tmpdir, "dummy.jpg"))
    os.rmdir(tmpdir)

    print("✅ PASSED: COCO bbox format correctly converted\n")


if __name__ == "__main__":
    test_positive_map_construction()
    test_loss_gradient_flow()
    test_empty_targets()
    test_all_empty_targets()
    test_hungarian_matching_quality()
    test_coco_bbox_format()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
