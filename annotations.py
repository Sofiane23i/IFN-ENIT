"""
Extract annotation bounding boxes from XML files and draw them on BMP images.
The annotations contain Segm_Area values representing horizontal x-coordinate ranges.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Installing Pillow...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'Pillow'])
    from PIL import Image, ImageDraw


def parse_segm_area(value: str) -> tuple:
    """
    Parse a Segm_Area value which can be:
    - A single number: "146" -> (146, 146)
    - A range: "53-65" -> (53, 65)
    """
    value = value.strip()
    if '-' in value:
        parts = value.split('-')
        return (int(parts[0]), int(parts[1]))
    else:
        return (int(value), int(value))


def parse_xml_annotations(xml_path: str) -> list:
    """
    Parse XML file and extract all Segm_Area bounding boxes.
    Returns list of (x_start, x_end) tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for record in root.findall('record'):
        # Find all Segm_Area elements
        for child in record:
            if child.tag.startswith('Segm_Area') and child.text:
                try:
                    bbox = parse_segm_area(child.text)
                    bboxes.append(bbox)
                except (ValueError, IndexError):
                    continue
    
    return bboxes


def draw_bboxes_on_image(image_path: str, bboxes: list, output_path: str, box_color='red', box_width=2):
    """
    Draw bounding boxes on an image and save to output path.
    Bboxes are (x_start, x_end) tuples - will span full image height.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    height = img.height
    
    for x_start, x_end in bboxes:
        # Ensure x0 <= x1
        x0, x1 = min(x_start, x_end), max(x_start, x_end)
        # Draw rectangle from top to bottom of image
        # Format: (x0, y0, x1, y1)
        draw.rectangle(
            [x0, 0, x1, height - 1],
            outline=box_color,
            width=box_width
        )
    
    img.save(output_path)
    return len(bboxes)


def process_dataset(xml_folder: str, bmp_folder: str, output_folder: str):
    """
    Process all XML files and draw annotations on corresponding BMP images.
    """
    xml_folder = Path(xml_folder)
    bmp_folder = Path(bmp_folder)
    output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    xml_files = list(xml_folder.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for xml_path in xml_files:
        # Get corresponding BMP file name from XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing {xml_path.name}: {e}")
            errors += 1
            continue
        
        # Try to get file_name from XML, or use XML filename
        file_name = None
        for record in root.findall('record'):
            name_elem = record.find('file_name')
            if name_elem is not None and name_elem.text:
                file_name = name_elem.text.strip()
                break
        
        if not file_name:
            file_name = xml_path.stem
        
        bmp_path = bmp_folder / f"{file_name}.bmp"
        output_path = output_folder / f"{file_name}.bmp"
        
        if not bmp_path.exists():
            print(f"Warning: BMP file not found: {bmp_path}")
            skipped += 1
            continue
        
        # Parse annotations
        bboxes = parse_xml_annotations(str(xml_path))
        
        if not bboxes:
            print(f"Warning: No annotations found in {xml_path.name}")
            skipped += 1
            continue
        
        # Draw and save
        num_boxes = draw_bboxes_on_image(str(bmp_path), bboxes, str(output_path))
        processed += 1
        
        if processed % 100 == 0:
            print(f"Processed {processed} images...")
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"Output saved to: {output_folder}")


if __name__ == '__main__':
    # Define paths
    base_path = Path(__file__).parent
    xml_folder = base_path / 'xml files'
    bmp_folder = base_path / 'Bmp files'
    output_folder = base_path / 'annotated_images'
    
    print(f"XML folder: {xml_folder}")
    print(f"BMP folder: {bmp_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    process_dataset(str(xml_folder), str(bmp_folder), str(output_folder))
