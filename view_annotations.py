"""
Display images with their Pascal VOC bounding box annotations.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import os

# Check if we have a display
HAS_DISPLAY = os.environ.get('DISPLAY') is not None or sys.platform == 'win32'

try:
    import matplotlib
    if not HAS_DISPLAY:
        matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except ImportError:
    USE_MATPLOTLIB = False


def parse_voc_annotation(xml_path: str) -> list:
    """
    Parse Pascal VOC XML annotation file.
    Returns list of (class_name, x_min, y_min, x_max, y_max) tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        boxes.append((class_name, x_min, y_min, x_max, y_max))
    
    return boxes


def display_image_with_boxes(image_path: str, boxes: list, box_color='red', box_width=2):
    """
    Display image with bounding boxes.
    """
    img = Image.open(image_path)
    
    # Convert to RGB if needed for display
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    
    for class_name, x_min, y_min, x_max, y_max in boxes:
        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_width)
        
        # Draw label
        label = class_name
        draw.text((x_min, max(0, y_min - 12)), label, fill=box_color)
    
    return img


def main():
    base_path = Path(__file__).parent
    voc_folder = base_path / 'bbox_annotations' / 'voc'
    images_folder = voc_folder / 'JPEGImages'
    annotations_folder = voc_folder / 'Annotations'
    output_folder = base_path / 'annotated_preview'
    output_folder.mkdir(exist_ok=True)
    
    # Get list of images
    image_files = sorted(images_folder.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {images_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Saving annotated images to: {output_folder}")
    print()
    
    # Process and save images
    for idx, img_path in enumerate(image_files):
        file_name = img_path.stem
        xml_path = annotations_folder / f"{file_name}.xml"
        
        if not xml_path.exists():
            print(f"Warning: No annotation for {file_name}")
            continue
        
        # Parse annotation
        boxes = parse_voc_annotation(str(xml_path))
        
        # Create annotated image
        img = display_image_with_boxes(str(img_path), boxes)
        
        # Save to file
        output_path = output_folder / f"{file_name}_annotated.jpg"
        img.save(output_path)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images...")
    
    print(f"\nDone! Annotated images saved to: {output_folder}")
    print(f"Open the folder to view images.")


if __name__ == '__main__':
    main()
