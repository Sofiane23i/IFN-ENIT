"""
Convert IFN-ENIT segmentation annotations to proper bounding boxes for object detection.
The XML files contain separator areas (vertical bars) BETWEEN characters.
Characters are located in the regions between these separator bars.

Outputs annotations in YOLO, COCO, and Pascal VOC formats.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np


def parse_segm_area(value: str) -> tuple:
    """
    Parse a Segm_Area value (separator region):
    - "146" -> (146, 146)
    - "53-65" -> (53, 65)
    """
    value = value.strip()
    if '-' in value:
        parts = value.split('-')
        return (int(parts[0]), int(parts[1]))
    else:
        v = int(value)
        return (v, v)


def parse_xml_separators(xml_path: str) -> list:
    """
    Parse XML file and extract all Segm_Area values as separator regions.
    Returns sorted list of (x_start, x_end) tuples representing separators.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    separators = []
    for record in root.findall('record'):
        for child in record:
            if child.tag.startswith('Segm_Area') and child.text:
                try:
                    sep = parse_segm_area(child.text)
                    # Ensure x_start <= x_end
                    separators.append((min(sep[0], sep[1]), max(sep[0], sep[1])))
                except (ValueError, IndexError):
                    continue
    
    # Sort by x position
    separators.sort(key=lambda x: x[0])
    return separators


def compute_character_regions(separators: list, img_width: int) -> list:
    """
    Compute character regions from separator positions.
    Characters are located BETWEEN the separator bars.
    
    Args:
        separators: List of (x_start, x_end) separator regions, sorted by x
        img_width: Image width
    
    Returns:
        List of (x_start, x_end) tuples for each character region
    """
    if not separators:
        return []
    
    char_regions = []
    
    # First character: from image start to first separator
    first_sep_start = separators[0][0]
    if first_sep_start > 0:
        char_regions.append((0, first_sep_start - 1))
    
    # Characters between separators
    for i in range(len(separators) - 1):
        sep_end = separators[i][1]
        next_sep_start = separators[i + 1][0]
        
        # Character region is between end of current separator and start of next
        if next_sep_start > sep_end + 1:
            char_regions.append((sep_end + 1, next_sep_start - 1))
    
    # Last character: from last separator to image end
    last_sep_end = separators[-1][1]
    if last_sep_end < img_width - 1:
        char_regions.append((last_sep_end + 1, img_width - 1))
    
    return char_regions


def find_vertical_bounds(img_array: np.ndarray, x_start: int, x_end: int, 
                         threshold: int = None, padding: int = 2) -> tuple:
    """
    Find the vertical (y) bounds of content within a horizontal strip.
    
    Args:
        img_array: Image as numpy array
        x_start, x_end: Horizontal range to analyze
        threshold: Pixel value threshold (auto-detected if None)
        padding: Padding to add around detected bounds
    
    Returns:
        (y_min, y_max) tuple, or None if no content found
    """
    height, width = img_array.shape[:2]
    
    # Clamp x coordinates to image bounds
    x_start = max(0, min(x_start, width - 1))
    x_end = max(0, min(x_end, width - 1))
    if x_start > x_end:
        x_start, x_end = x_end, x_start
    
    # Extract the vertical strip
    strip = img_array[:, x_start:x_end+1]
    
    # Handle color/palette images - convert to grayscale
    if len(strip.shape) == 3:
        # Convert to grayscale using luminance formula
        strip = 0.299 * strip[:,:,0] + 0.587 * strip[:,:,1] + 0.114 * strip[:,:,2]
    
    strip = strip.astype(np.float32)
    
    # Auto-detect threshold using Otsu-like method
    if threshold is None:
        # Find background (most common value, usually brightest)
        bg_value = np.percentile(strip, 90)  # Top 90% is likely background
        fg_value = np.percentile(strip, 10)  # Bottom 10% is likely foreground
        threshold = (bg_value + fg_value) / 2
    
    # Determine if dark text on light background or vice versa
    mean_val = np.mean(strip)
    if mean_val > 127:
        # Light background, dark text - find pixels darker than threshold
        row_has_content = np.any(strip < threshold, axis=1)
    else:
        # Dark background, light text - find pixels brighter than threshold
        row_has_content = np.any(strip > threshold, axis=1)
    
    content_rows = np.where(row_has_content)[0]
    
    if len(content_rows) == 0:
        return None
    
    y_min = max(0, content_rows[0] - padding)
    y_max = min(height - 1, content_rows[-1] + padding)
    
    return (y_min, y_max)


def convert_to_yolo(bbox: tuple, img_width: int, img_height: int, class_id: int = 0) -> str:
    """
    Convert bbox to YOLO format: class_id x_center y_center width height (normalized)
    bbox: (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_to_voc(bbox: tuple, filename: str, img_width: int, img_height: int, 
                   class_name: str = "X") -> str:
    """
    Convert bbox to Pascal VOC XML format.
    bbox: (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return f"""    <object>
        <name>{class_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{x_min}</xmin>
            <ymin>{y_min}</ymin>
            <xmax>{x_max}</xmax>
            <ymax>{y_max}</ymax>
        </bndbox>
    </object>"""


def create_voc_annotation(filename: str, img_width: int, img_height: int, 
                          objects_xml: list) -> str:
    """Create full Pascal VOC annotation XML."""
    objects_str = "\n".join(objects_xml)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>images</folder>
    <filename>{filename}</filename>
    <size>
        <width>{img_width}</width>
        <height>{img_height}</height>
        <depth>3</depth>
    </size>
{objects_str}
</annotation>"""


def process_dataset(xml_folder: str, bmp_folder: str, output_folder: str,
                    output_format: str = 'all', threshold: int = None, padding: int = 2):
    """
    Process all XML files and convert to object detection format.
    
    Args:
        xml_folder: Path to folder with XML annotations
        bmp_folder: Path to folder with BMP images
        output_folder: Path to save converted annotations
        output_format: 'yolo', 'coco', 'voc', or 'all'
        threshold: Pixel threshold for content detection (None for auto-detect)
        padding: Padding around detected content
    """
    xml_folder = Path(xml_folder)
    bmp_folder = Path(bmp_folder)
    output_folder = Path(output_folder)
    
    # Create output directories
    if output_format in ['yolo', 'all']:
        (output_folder / 'yolo' / 'labels').mkdir(parents=True, exist_ok=True)
        (output_folder / 'yolo' / 'images').mkdir(parents=True, exist_ok=True)
    if output_format in ['voc', 'all']:
        (output_folder / 'voc' / 'Annotations').mkdir(parents=True, exist_ok=True)
        (output_folder / 'voc' / 'JPEGImages').mkdir(parents=True, exist_ok=True)
    if output_format in ['coco', 'all']:
        (output_folder / 'coco').mkdir(parents=True, exist_ok=True)
    
    # COCO format structures
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "X", "supercategory": "text"}]
    }
    annotation_id = 0
    image_id = 0
    
    xml_files = list(xml_folder.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    processed = 0
    skipped = 0
    total_boxes = 0
    
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing {xml_path.name}: {e}")
            skipped += 1
            continue
        
        # Get file name
        file_name = None
        for record in root.findall('record'):
            name_elem = record.find('file_name')
            if name_elem is not None and name_elem.text:
                file_name = name_elem.text.strip()
                break
        
        if not file_name:
            file_name = xml_path.stem
        
        bmp_path = bmp_folder / f"{file_name}.bmp"
        
        if not bmp_path.exists():
            skipped += 1
            continue
        
        # Load image
        img = Image.open(bmp_path)
        # Convert to RGB to handle palette images properly
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        img_array = np.array(img_rgb)
        img_width, img_height = img.size
        
        # Parse separator regions from XML
        separators = parse_xml_separators(str(xml_path))
        
        if not separators:
            skipped += 1
            continue
        
        # Compute character regions (between separators)
        char_regions = compute_character_regions(separators, img_width)
        
        if not char_regions:
            skipped += 1
            continue
        
        # Convert character regions to proper bboxes with vertical bounds
        bboxes = []
        for x_start, x_end in char_regions:
            # Find vertical bounds for this character region
            y_bounds = find_vertical_bounds(img_array, x_start, x_end, threshold, padding)
            
            if y_bounds is None:
                continue
            
            y_min, y_max = y_bounds
            bboxes.append((x_start, y_min, x_end, y_max))
        
        if not bboxes:
            skipped += 1
            continue
        
        total_boxes += len(bboxes)
        
        # Save in requested formats (img_rgb is already RGB)
        if output_format in ['yolo', 'all']:
            # Copy/convert image
            img_rgb.save(output_folder / 'yolo' / 'images' / f"{file_name}.jpg")
            
            # Save labels
            yolo_lines = [convert_to_yolo(bbox, img_width, img_height) for bbox in bboxes]
            with open(output_folder / 'yolo' / 'labels' / f"{file_name}.txt", 'w') as f:
                f.write("\n".join(yolo_lines))
        
        if output_format in ['voc', 'all']:
            # Copy/convert image
            img_rgb.save(output_folder / 'voc' / 'JPEGImages' / f"{file_name}.jpg")
            
            # Save annotation
            objects_xml = [convert_to_voc(bbox, f"{file_name}.jpg", img_width, img_height) 
                          for bbox in bboxes]
            voc_xml = create_voc_annotation(f"{file_name}.jpg", img_width, img_height, objects_xml)
            with open(output_folder / 'voc' / 'Annotations' / f"{file_name}.xml", 'w') as f:
                f.write(voc_xml)
        
        if output_format in ['coco', 'all']:
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{file_name}.bmp",
                "width": img_width,
                "height": img_height
            })
            
            # Add annotations
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                # Convert to native Python int for JSON serialization
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO: [x, y, w, h]
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                })
                annotation_id += 1
            
            image_id += 1
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} images, {total_boxes} boxes...")
    
    # Save COCO annotations
    if output_format in ['coco', 'all']:
        with open(output_folder / 'coco' / 'annotations.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    # Create YOLO classes file
    if output_format in ['yolo', 'all']:
        with open(output_folder / 'yolo' / 'classes.txt', 'w') as f:
            f.write("X\n")
    
    print(f"\nDone!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Output saved to: {output_folder}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert IFN-ENIT annotations to object detection format')
    parser.add_argument('--format', choices=['yolo', 'coco', 'voc', 'all'], default='all',
                       help='Output format (default: all)')
    parser.add_argument('--threshold', type=int, default=None,
                       help='Pixel threshold for content detection (default: auto-detect)')
    parser.add_argument('--padding', type=int, default=2,
                       help='Padding around detected content (default: 2)')
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    xml_folder = base_path / 'xml files'
    bmp_folder = base_path / 'Bmp files'
    output_folder = base_path / 'bbox_annotations'
    
    print(f"XML folder: {xml_folder}")
    print(f"BMP folder: {bmp_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Output format: {args.format}")
    print(f"Threshold: {args.threshold if args.threshold else 'auto-detect'}")
    print(f"Padding: {args.padding}")
    print()
    
    process_dataset(
        str(xml_folder), 
        str(bmp_folder), 
        str(output_folder),
        output_format=args.format,
        threshold=args.threshold,
        padding=args.padding
    )
