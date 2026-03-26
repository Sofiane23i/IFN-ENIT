"""Evaluate a trained detection model and produce visual reports.

Generates:
  1. Train / Val loss curves (read from TensorBoard logs)
  2. Precision-Recall curve
  3. Confusion matrix (detected vs missed)
  4. Per-image detection examples
  5. Prints AP, precision, recall, F1 at the chosen IoU & score thresholds

Usage:
    python evaluate_model.py --model fasterrcnn
    python evaluate_model.py --model ssd --iou-thresh 0.3 --score-thresh 0.4
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.ops import box_iou
from torch.utils.data import DataLoader, random_split

# Reuse dataset & helpers from training script
from train_detection import IFNENITDetection, get_transform, collate_fn

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / 'eval_results'


# ── 1. Read TensorBoard logs and plot train/val curves ────────────────────────

def plot_train_val_curves(log_dir: Path, out_dir: Path):
    """Read TensorBoard event files and plot Loss/train & Loss/val."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("WARNING: tensorboard not installed, skipping loss curves.")
        return

    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    train_tag = 'Loss/train'
    val_tag = 'Loss/val'

    if train_tag not in available_tags and val_tag not in available_tags:
        print(f"WARNING: No Loss/train or Loss/val tags in {log_dir}")
        print(f"  Available tags: {available_tags}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if train_tag in available_tags:
        events = ea.Scalars(train_tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        ax.plot(steps, vals, label='Train Loss', color='tab:blue', linewidth=2)

    if val_tag in available_tags:
        events = ea.Scalars(val_tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        ax.plot(steps, vals, label='Val Loss', color='tab:orange', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'loss_curves.png', dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curves.png")


# ── 2. Run inference on val set and collect predictions ───────────────────────

@torch.no_grad()
def collect_predictions(model, data_loader, device, score_thresh):
    """Run model in eval mode and collect per-image predictions + ground truth."""
    model.eval()
    all_preds = []   # list of dicts: {boxes, scores}
    all_targets = [] # list of dicts: {boxes}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            keep = out['scores'] >= score_thresh
            all_preds.append({
                'boxes': out['boxes'][keep].cpu(),
                'scores': out['scores'][keep].cpu(),
            })
            all_targets.append({
                'boxes': tgt['boxes'],
            })

    return all_preds, all_targets


# ── 3. Match predictions to ground truth boxes ───────────────────────────────

def match_detections(preds, targets, iou_thresh):
    """For each image, match predicted boxes to GT boxes and return TP/FP/FN counts.

    Returns:
        all_scores: confidence scores of all predictions
        all_tp: boolean array (True if prediction is a TP)
        total_gt: total number of ground-truth boxes
        per_image: list of dicts with tp/fp/fn per image
    """
    all_scores = []
    all_tp = []
    total_gt = 0
    per_image = []

    for pred, tgt in zip(preds, targets):
        gt_boxes = tgt['boxes']
        pred_boxes = pred['boxes']
        scores = pred['scores']
        n_gt = len(gt_boxes)
        total_gt += n_gt

        tp = 0
        fp = 0
        matched_gt = set()

        if len(pred_boxes) > 0 and n_gt > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            # Sort predictions by descending confidence
            order = scores.argsort(descending=True)
            for idx in order:
                best_iou, best_gt = ious[idx].max(0)
                if best_iou >= iou_thresh and best_gt.item() not in matched_gt:
                    matched_gt.add(best_gt.item())
                    all_tp.append(True)
                    tp += 1
                else:
                    all_tp.append(False)
                    fp += 1
                all_scores.append(scores[idx].item())
        elif len(pred_boxes) > 0:
            # All predictions are FP (no GT)
            for s in scores:
                all_scores.append(s.item())
                all_tp.append(False)
                fp += 1

        fn = n_gt - tp
        per_image.append({'tp': tp, 'fp': fp, 'fn': fn, 'n_gt': n_gt, 'n_pred': len(pred_boxes)})

    return np.array(all_scores), np.array(all_tp), total_gt, per_image


# ── 4. Compute AP from scores + TP labels ────────────────────────────────────

def compute_ap(scores, tp, total_gt):
    """Compute Average Precision using all-point interpolation."""
    if total_gt == 0 or len(scores) == 0:
        return 0.0, np.array([]), np.array([])

    order = np.argsort(-scores)
    tp_sorted = tp[order].astype(float)
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)

    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp)

    # All-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap, precision, recall


# ── 5. Plot Precision-Recall curve ────────────────────────────────────────────

def plot_pr_curve(precision, recall, ap, iou_thresh, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where='post', color='tab:blue', linewidth=2)
    ax.fill_between(recall, precision, step='post', alpha=0.15, color='tab:blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve  (AP@IoU={iou_thresh:.2f} = {ap:.3f})')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'precision_recall.png', dpi=150)
    plt.close(fig)
    print(f"  Saved precision_recall.png")


# ── 6. Confusion matrix (TP / FP / FN) ───────────────────────────────────────

def plot_confusion_matrix(per_image, out_dir):
    total_tp = sum(d['tp'] for d in per_image)
    total_fp = sum(d['fp'] for d in per_image)
    total_fn = sum(d['fn'] for d in per_image)

    # 2x2: rows = actual (GT present / GT absent), cols = predicted (detected / not detected)
    # For single-class detection:
    #   TP: GT box matched by prediction
    #   FP: prediction with no matching GT
    #   FN: GT box missed (no matching prediction)
    #   TN: not meaningful for object detection
    matrix = np.array([[total_tp, total_fn],
                        [total_fp, 0]], dtype=int)
    labels = ['Character', 'Background']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Detected\n(Positive)', 'Missed\n(Negative)'])
    ax.set_yticklabels(['GT Present\n(Actual Pos)', 'No GT\n(Actual Neg)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Detection Confusion Matrix')

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            label = ''
            if i == 0 and j == 0:
                label = f'TP\n{val}'
            elif i == 0 and j == 1:
                label = f'FN\n{val}'
            elif i == 1 and j == 0:
                label = f'FP\n{val}'
            else:
                label = f'TN\nN/A'
            text_color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix.png")


# ── 7. TP/FP/FN distribution histogram ───────────────────────────────────────

def plot_detection_histogram(per_image, out_dir):
    tp_counts = [d['tp'] for d in per_image]
    fp_counts = [d['fp'] for d in per_image]
    fn_counts = [d['fn'] for d in per_image]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title, color in zip(axes,
                                       [tp_counts, fp_counts, fn_counts],
                                       ['True Positives', 'False Positives', 'False Negatives'],
                                       ['tab:green', 'tab:red', 'tab:orange']):
        ax.hist(data, bins=max(max(data, default=0), 1), color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Count per image')
        ax.set_ylabel('Number of images')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Per-image Detection Distribution', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'detection_histogram.png', dpi=150)
    plt.close(fig)
    print(f"  Saved detection_histogram.png")


# ── 8. Score distribution for TP vs FP ────────────────────────────────────────

def plot_score_distribution(scores, tp, out_dir):
    if len(scores) == 0:
        return
    tp_scores = scores[tp]
    fp_scores = scores[~tp]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    if len(tp_scores) > 0:
        ax.hist(tp_scores, bins=bins, alpha=0.6, label=f'TP ({len(tp_scores)})', color='tab:green', edgecolor='black')
    if len(fp_scores) > 0:
        ax.hist(fp_scores, bins=bins, alpha=0.6, label=f'FP ({len(fp_scores)})', color='tab:red', edgecolor='black')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution: TP vs FP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'score_distribution.png', dpi=150)
    plt.close(fig)
    print(f"  Saved score_distribution.png")


# ── 9. Sample detection visualizations ────────────────────────────────────────

def plot_sample_detections(model, dataset, device, score_thresh, out_dir, n_samples=8):
    """Draw GT (green) and predictions (red) on a grid of sample images."""
    model.eval()
    to_tensor = transforms.ToTensor()
    n_samples = min(n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax_idx, data_idx in enumerate(indices):
        img_tensor, target = dataset[data_idx]
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            inp = img_tensor
        else:
            img_np = np.array(img_tensor) / 255.0
            inp = to_tensor(img_tensor)

        with torch.no_grad():
            out = model([inp.to(device)])[0]

        ax = axes[ax_idx]
        ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)

        # Ground truth boxes in green
        for box in target['boxes']:
            x1, y1, x2, y2 = box.tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        # Predictions in red
        keep = out['scores'] >= score_thresh
        for box, score in zip(out['boxes'][keep], out['scores'][keep]):
            x1, y1, x2, y2 = box.cpu().tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, f'{score:.2f}', fontsize=7, color='red',
                    bbox=dict(facecolor='white', alpha=0.6, pad=0.5))

        ax.set_title(f'Image {data_idx}', fontsize=9)
        ax.axis('off')

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Sample Detections  (green=GT, red=pred)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'sample_detections.png', dpi=150)
    plt.close(fig)
    print(f"  Saved sample_detections.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate detection model & generate reports")
    parser.add_argument('--model', choices=['fasterrcnn', 'ssd'], default='fasterrcnn')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pth checkpoint (default: checkpoints/<model>_final.pth)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--score-thresh', type=float, default=0.5,
                        help='Confidence threshold for predictions (default: 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths
    ann_file = BASE_DIR / 'bbox_annotations' / 'coco' / 'annotations.json'
    img_dir = BASE_DIR / 'Bmp files'
    ckpt_path = Path(args.checkpoint) if args.checkpoint else BASE_DIR / 'checkpoints' / f'{args.model}_final.pth'

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Train a model first:  python train_detection.py")
        return

    OUT_DIR.mkdir(exist_ok=True)
    print(f'Output directory: {OUT_DIR}\n')

    # ── Loss curves from TensorBoard logs ──
    print("[1/6] Plotting train/val loss curves...")
    log_dir = BASE_DIR / 'runs' / args.model
    if log_dir.exists():
        plot_train_val_curves(log_dir, OUT_DIR)
    else:
        print(f"  WARNING: No TensorBoard logs at {log_dir}, skipping.")

    # ── Load model ──
    print(f"\n[2/6] Loading {args.model} from {ckpt_path}...")
    if args.model == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    else:
        model = ssd300_vgg16(num_classes=2)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("  Model loaded.")

    # ── Prepare val dataset (same split as training) ──
    full_dataset = IFNENITDetection(
        str(ann_file), str(img_dir), transform=get_transform(train=False))
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    print(f"  Validation set: {val_size} images")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # ── Run inference ──
    print(f"\n[3/6] Running inference (score_thresh={args.score_thresh}, iou_thresh={args.iou_thresh})...")
    preds, targets = collect_predictions(model, val_loader, device, args.score_thresh)

    # ── Compute metrics ──
    print("\n[4/6] Computing metrics...")
    scores, tp, total_gt, per_image = match_detections(preds, targets, args.iou_thresh)
    ap, precision, recall = compute_ap(scores, tp, total_gt)

    total_tp = int(tp.sum()) if len(tp) > 0 else 0
    total_fp = int((~tp).sum()) if len(tp) > 0 else 0
    total_fn = total_gt - total_tp

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS  (IoU={args.iou_thresh}, score>={args.score_thresh})")
    print(f"{'='*50}")
    print(f"  Total GT boxes   : {total_gt}")
    print(f"  Total predictions : {total_tp + total_fp}")
    print(f"  True Positives    : {total_tp}")
    print(f"  False Positives   : {total_fp}")
    print(f"  False Negatives   : {total_fn}")
    print(f"  ─────────────────────────────")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  AP@IoU={args.iou_thresh:.2f}       : {ap:.4f}")
    print(f"{'='*50}\n")

    # ── Generate all plots ──
    print("[5/6] Generating plots...")
    if len(precision) > 0:
        plot_pr_curve(precision, recall, ap, args.iou_thresh, OUT_DIR)
    plot_confusion_matrix(per_image, OUT_DIR)
    plot_detection_histogram(per_image, OUT_DIR)
    plot_score_distribution(scores, tp, OUT_DIR)

    print("\n[6/6] Generating sample detection visualizations...")
    plot_sample_detections(model, val_dataset, device, args.score_thresh, OUT_DIR)

    print(f"\nDone! All results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
