"""Training scripts for character object detection using Faster R-CNN and SSD.

This file provides utilities to load the COCO-format dataset produced by
`convert_to_bbox.py` and to train two torchvision models:
  * Faster R-CNN with a ResNet50-FPN backbone
  * SSD300 with a VGG16 backbone

Usage examples:
    python train_detection.py --model fasterrcnn
    python train_detection.py --model ssd --epochs 20 --lr 0.001

The COCO annotations live at bbox_annotations/coco/annotations.json
and images are loaded from "Bmp files/".

Category id 0 in annotations is remapped to 1 (torchvision reserves 0 for background).
TensorBoard logs are written to runs/<model>/ — launch with:
    tensorboard --logdir runs
"""

import argparse
import os
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import json
import torch.nn as nn


BASE_DIR = Path(__file__).parent


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2 conv layers + skip connection)."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class LightHTRBackbone(nn.Module):
    """Mini-ResNet + FPN backbone for offline HTR images.

    Same structure as ResNet-50-FPN but with only 1 BasicBlock per stage
    instead of [3,4,6,3] Bottleneck blocks.  Much lighter for simple
    black-on-white handwriting images.

    Architecture:
        Stem:   7×7 conv, stride 2, BN, ReLU, MaxPool  → stride 4
        Layer1: 1 BasicBlock (64 ch)                    → stride 4  (C2)
        Layer2: 1 BasicBlock (128 ch, stride 2)         → stride 8  (C3)
        Layer3: 1 BasicBlock (256 ch, stride 2)         → stride 16 (C4)
        Layer4: 1 BasicBlock (256 ch, stride 2)         → stride 32 (C5)
        FPN:    top-down + lateral on C3, C4, C5         → P3, P4, P5
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Stem (same as ResNet)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual stages — 1 block each (vs ResNet-50's [3,4,6,3])
        self.layer1 = BasicBlock(64, 64)        # stride 4, C2
        self.layer2 = BasicBlock(64, 128, 2)    # stride 8, C3
        self.layer3 = BasicBlock(128, 256, 2)   # stride 16, C4
        self.layer4 = BasicBlock(256, 256, 2)   # stride 32, C5

        # FPN lateral connections
        fpn_ch = 128
        self.lateral3 = nn.Conv2d(128, fpn_ch, 1)
        self.lateral4 = nn.Conv2d(256, fpn_ch, 1)
        self.lateral5 = nn.Conv2d(256, fpn_ch, 1)

        # FPN smooth layers
        self.smooth3 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)

        self.out_channels = fpn_ch

        # Weight initialization (same as ResNet)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Residual stages
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN top-down
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + nn.functional.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lateral3(c3) + nn.functional.interpolate(p4, size=c3.shape[2:], mode='nearest')

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return {'0': p3, '1': p4, '2': p5}


def build_fasterrcnn_light(num_classes=2):
    """Faster R-CNN with the lightweight 5-layer FPN backbone."""
    backbone = LightHTRBackbone(in_channels=3)
    # Multi-scale anchors matching the 3 FPN levels (strides 8, 16, 32)
    anchor_generator = AnchorGenerator(
        sizes=((8, 16), (32, 64), (64, 128)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )
    from torchvision.ops import MultiScaleRoIAlign
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2'], output_size=7, sampling_ratio=2
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=200,
        max_size=800,
    )
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


class IFNENITDetection(Dataset):
    """Custom dataset that loads BMP images and COCO-format annotations."""

    def __init__(self, ann_file: str, img_dir: str, transform=None):
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.img_dir = Path(img_dir)
        self.transform = transform

        # Build lookup: image_id -> image info
        self.images = {img['id']: img for img in coco['images']}

        # Build lookup: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in coco['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

        # Only keep images that have at least one annotation
        self.image_ids = [img_id for img_id in self.images
                          if img_id in self.img_to_anns]
        print(f"Dataset: {len(self.image_ids)} images with annotations "
              f"({len(coco['annotations'])} total boxes)")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.img_dir / img_info['file_name']

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        anns = self.img_to_anns[img_id]
        boxes = []
        for a in anns:
            x, y, w, h = a['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Remap category 0 -> 1 (torchvision reserves 0 for background)
        labels = torch.ones(len(boxes), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
        }
        return img, target


def get_transform(train: bool):
    t = [transforms.ToTensor()]
    if train:
        t.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(t)


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    global_step = (epoch - 1) * len(data_loader)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_loss = losses.item()
        epoch_loss += batch_loss
        n_batches += 1
        global_step += 1

        # Log per-batch losses to TensorBoard
        writer.add_scalar('BatchLoss/total', batch_loss, global_step)
        for name, v in loss_dict.items():
            writer.add_scalar(f'BatchLoss/{name}', v.item(), global_step)

    avg_loss = epoch_loss / max(n_batches, 1)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    print(f"Epoch {epoch}  train loss: {avg_loss:.4f}", end='')
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, writer):
    model.train()  # keep in train mode so loss is computed
    val_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        val_loss += losses.item()
        n_batches += 1

    avg_loss = val_loss / max(n_batches, 1)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    print(f"  val loss: {avg_loss:.4f}", end='')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train detection model on IFN-ENIT")
    parser.add_argument('--model', choices=['fasterrcnn', 'fasterrcnn_light', 'ssd'],
                        default='fasterrcnn',
                        help='Architecture (default: fasterrcnn, fasterrcnn_light = 5-layer HTR backbone)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths
    ann_file = BASE_DIR / 'bbox_annotations' / 'coco' / 'annotations.json'
    img_dir = BASE_DIR / 'Bmp files'

    if not ann_file.exists():
        print(f"ERROR: annotations not found at {ann_file}")
        print("Run  python convert_to_bbox.py --format coco  first.")
        return
    if not img_dir.exists():
        print(f"ERROR: image folder not found at {img_dir}")
        return

    # Dataset & train/val split
    full_dataset = IFNENITDetection(
        str(ann_file), str(img_dir), transform=get_transform(train=True))
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    print(f"Split: {train_size} train / {val_size} val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # Model — num_classes = 1 foreground + 1 background = 2
    if args.model == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    elif args.model == 'fasterrcnn_light':
        model = build_fasterrcnn_light(num_classes=2)
    else:
        model = ssd300_vgg16(num_classes=2)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {args.model}  ({n_params:.1f}M parameters)')

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TensorBoard — single writer, Loss/train and Loss/val on the same chart
    log_dir = BASE_DIR / 'runs' / args.model
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f'TensorBoard logs: {log_dir}')
    print(f'Launch dashboard:  tensorboard --logdir runs\n')

    ckpt_dir = BASE_DIR / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer)
        val_loss = evaluate(model, val_loader, device, epoch, writer)
        writer.flush()
        lr_scheduler.step()
        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)", end='')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(ckpt_dir / f'{args.model}_best.pth'))
            print(f"  [best saved]", end='')

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), str(ckpt_dir / f'{args.model}_epoch{epoch}.pth'))
            print(f"  [ckpt saved]", end='')

        print()

    writer.close()

    # Save final model
    ckpt_path = ckpt_dir / f'{args.model}_final.pth'
    torch.save(model.state_dict(), str(ckpt_path))
    print(f'\nTraining completed. Model saved to {ckpt_path}')


if __name__ == '__main__':
    main()
