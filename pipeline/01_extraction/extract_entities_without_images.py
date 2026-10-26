#!/usr/bin/env python3
"""
Extract Entities Without Images - Copy Script

This script scans the fetched entities directory and copies all entities
that have zero images to a separate directory structure.

The original entities remain untouched in the source directory.

Features:
- Scans entities directory for entity_data.json files
- Identifies entities with zero images (has_images: false OR total_images: 0)
- Copies entire entity directory (JSON + any files) to new location
- Maintains same directory structure
- Creates summary statistics
- Preserves original data (non-destructive)

Usage:
    python3 extract_entities_without_images.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))


def find_all_entity_json_files(entities_dir: str) -> List[str]:
    """
    Find all entity_data.json files in the entities directory.

    Args:
        entities_dir: Path to the entities directory

    Returns:
        List of paths to entity_data.json files
    """
    json_files = []
    entities_path = Path(entities_dir)

    if not entities_path.exists():
        print(f"[ERROR] Directory not found: {entities_dir}")
        return []

    # Walk through all subdirectories
    for entity_dir in entities_path.iterdir():
        if entity_dir.is_dir():
            entity_json = entity_dir / "entity_data.json"
            if entity_json.exists():
                json_files.append(str(entity_json))

    return json_files


def load_entity_data(json_file_path: str) -> Dict:
    """
    Load entity data from JSON file.

    Args:
        json_file_path: Path to entity_data.json file

    Returns:
        Dictionary with entity data, or None if failed
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load {json_file_path}: {e}")
        return None


def count_images_in_directory(entity_dir: Path) -> int:
    """
    Count actual image files in an entity directory.

    Args:
        entity_dir: Path to entity directory

    Returns:
        Number of image files found
    """
    image_count = 0
    for file in entity_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_count += 1
    return image_count


def has_no_images(entity_data: Dict, entity_dir: Path) -> bool:
    """
    Check if entity has no images.

    Checks both JSON metadata and actual files in directory.

    Args:
        entity_data: Entity data dictionary from JSON
        entity_dir: Path to entity directory

    Returns:
        True if entity has no images, False otherwise
    """
    # Check JSON metadata
    has_images_flag = entity_data.get('has_images', True)
    total_images = entity_data.get('total_images', 0)
    images_list = entity_data.get('images', [])

    # Check actual files
    actual_image_count = count_images_in_directory(entity_dir)

    # Entity has no images if:
    # 1. has_images flag is False
    # 2. total_images is 0
    # 3. images list is empty
    # 4. actual file count is 0
    return (not has_images_flag or
            total_images == 0 or
            len(images_list) == 0 or
            actual_image_count == 0)


def copy_entity_directory(source_dir: Path, dest_dir: Path) -> bool:
    """
    Copy entire entity directory to destination.

    Args:
        source_dir: Source entity directory
        dest_dir: Destination entity directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if needed
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy entire directory
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy {source_dir} to {dest_dir}: {e}")
        return False


def extract_entities_without_images(input_dir: str, output_dir: str):
    """
    Extract all entities without images to a separate directory.

    Args:
        input_dir: Input directory containing entities/ subdirectory
        output_dir: Output directory for entities without images
    """
    print("="*80)
    print("Extract Entities Without Images - Copy Script")
    print("="*80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Locate entities directories
    input_entities_dir = os.path.join(input_dir, "entities")
    output_entities_dir = os.path.join(output_dir, "entities")

    if not os.path.exists(input_entities_dir):
        print(f"[ERROR] Input entities directory not found: {input_entities_dir}")
        return

    # Create output directory
    os.makedirs(output_entities_dir, exist_ok=True)

    # Find all entity JSON files
    print("Scanning for entities...\n")
    json_files = find_all_entity_json_files(input_entities_dir)

    if not json_files:
        print("[ERROR] No entity JSON files found!")
        return

    print(f"Found {len(json_files)} entities\n")
    print("Processing entities...\n")

    # Statistics
    total_entities = 0
    entities_without_images = 0
    entities_with_images = 0
    copied_successfully = 0
    copy_failed = 0
    load_failed = 0

    entities_without_images_list = []
    entities_with_images_list = []

    # Process each entity
    for json_file in json_files:
        entity_data = load_entity_data(json_file)

        if entity_data is None:
            load_failed += 1
            continue

        total_entities += 1
        entity_title = entity_data.get('title', 'Unknown')
        entity_dir = Path(json_file).parent

        # Check if entity has no images
        if has_no_images(entity_data, entity_dir):
            entities_without_images += 1
            entities_without_images_list.append(entity_title)

            # Copy entity directory
            entity_dir_name = entity_dir.name
            dest_entity_dir = Path(output_entities_dir) / entity_dir_name

            if copy_entity_directory(entity_dir, dest_entity_dir):
                copied_successfully += 1
                print(f"  ✓ Copied: {entity_title}")
            else:
                copy_failed += 1
                print(f"  ✗ Failed: {entity_title}")
        else:
            entities_with_images += 1
            entities_with_images_list.append(entity_title)

    # Print summary
    print()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print()
    print(f"📊 Total entities scanned:       {total_entities:,}")
    print(f"📊 Entities with images:         {entities_with_images:,}")
    print(f"📊 Entities without images:      {entities_without_images:,}")
    print(f"✓  Successfully copied:          {copied_successfully:,}")
    print(f"✗  Copy failed:                  {copy_failed:,}")
    if load_failed > 0:
        print(f"⚠️  Failed to load:              {load_failed:,}")
    print()

    # Calculate percentages
    if total_entities > 0:
        pct_without_images = (entities_without_images / total_entities) * 100
        pct_with_images = (entities_with_images / total_entities) * 100
        print(f"📈 Entities without images:      {pct_without_images:.1f}%")
        print(f"📈 Entities with images:         {pct_with_images:.1f}%")
        print()

    print("="*80)

    # Save statistics
    stats = {
        'total_entities_scanned': total_entities,
        'entities_with_images': entities_with_images,
        'entities_without_images': entities_without_images,
        'copied_successfully': copied_successfully,
        'copy_failed': copy_failed,
        'load_failed': load_failed,
        'percentage_without_images': (entities_without_images / total_entities * 100) if total_entities > 0 else 0,
        'entities_without_images_list': entities_without_images_list,
        'entities_with_images_list': entities_with_images_list
    }

    stats_file = os.path.join(output_dir, "extraction_statistics.json")
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Statistics saved to: {stats_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save statistics: {e}")

    # Save entity lists
    try:
        # List of entities without images
        without_images_file = os.path.join(output_dir, "entities_without_images_list.txt")
        with open(without_images_file, 'w', encoding='utf-8') as f:
            for entity in sorted(entities_without_images_list):
                f.write(f"{entity}\n")
        print(f"✓ List of entities without images saved to: {without_images_file}")

        # List of entities with images (for reference)
        with_images_file = os.path.join(output_dir, "entities_with_images_list.txt")
        with open(with_images_file, 'w', encoding='utf-8') as f:
            for entity in sorted(entities_with_images_list):
                f.write(f"{entity}\n")
        print(f"✓ List of entities with images saved to: {with_images_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save entity lists: {e}")

    print()
    print("="*80)
    print("Extraction complete!")
    print("="*80)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract entities without images to a separate directory'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(DATA / '01_extraction' / 'newdata' / 'temporal_relevant'),
        help='Input directory containing the entity subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DATA / '01_extraction' / 'newdata' / 'no_images'),
        help='Output directory for entities without images'
    )

    args = parser.parse_args()

    extract_entities_without_images(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()