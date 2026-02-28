"""Training scripts for character object detection using Faster R-CNN and SSD.

This file provides utilities to load the COCO-format dataset produced by
`convert_to_bbox.py` and to train two torchvision models:
  * Faster R-CNN with a ResNet50-FPN backbone
  * SSD300 with a MobileNetV3-Large backbone

Usage examples:
    python train_detection.py --model fasterrcnn --data-dir bbox_annotations/coco
    python train_detection.py --model ssd --data-dir bbox_annotations/coco

The script assumes the COCO annotations live at
    <data_dir>/annotations.json
and the images in the same directory or `images/` subfolder.

Only a single class (category id 0) is used. The dataset loader filters
out any annotations with area <= 0.  
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.datasets import CocoDetection


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train: bool):
    transformations = []
    transformations.append(transforms.ToTensor())
    if train:
        # add whatever augmentation you need
        transformations.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transformations)


def make_dataset(data_dir: str, train: bool):
    ann_file = os.path.join(data_dir, "annotations.json")
    img_dir = data_dir
    if not os.path.isdir(img_dir):
        # maybe images are under images/
        img_dir = os.path.join(data_dir, "images")
    dataset = CocoDetection(img_dir, ann_file, transforms=get_transform(train))
    return dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        # convert targets to expected format (list of dicts)
        new_targets = []
        for t in targets:
            boxes = torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32)
            # COCO bbox is [x,y,w,h]; convert to x1,y1,x2,y2
            boxes[:, 2:] += boxes[:, :2]
            labels = torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64)
            new_targets.append({'boxes': boxes.to(device), 'labels': labels.to(device)})
        loss_dict = model(images, new_targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch} training loss: {losses.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument('--model', choices=['fasterrcnn', 'ssd'], required=True,
                        help='Which architecture to train')
    parser.add_argument('--data-dir', type=str, default='bbox_annotations/coco',
                        help='Directory containing COCO annotations and images')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    train_dataset = make_dataset(args.data_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)

    # create model
    if args.model == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(num_classes=1)
    else:
        model = ssd300_vgg16(num_classes=1)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)

    for epoch in range(1, args.epochs+1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        # you could add validation here

    # save final model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{args.model}_final.pth')
    print('Training completed, model saved to checkpoints/')


if __name__ == '__main__':
    main()
