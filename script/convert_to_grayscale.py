"""
Convert distilled images from color to grayscale.

Usage:
    python -m script.convert_to_grayscale train_img/mnist/png/step001000.png
    python -m script.convert_to_grayscale train_img/mnist/png --all
"""

import os
import glob
import argparse
from PIL import Image


def convert_to_grayscale(path: str, output: str = None, convert_all: bool = False):
    """
    Convert image(s) to grayscale.
    
    Args:
        path: Path to image file or directory
        output: Output path (optional, overwrites if not specified)
        convert_all: If True and path is a directory, convert all PNG files
    """
    if os.path.isdir(path):
        if convert_all:
            files = glob.glob(os.path.join(path, '*.png'))
            print(f"Converting {len(files)} files...")
            for f in files:
                _convert_single(f)
            print("Done!")
        else:
            print("Path is a directory. Use --all to convert all PNG files.")
    else:
        _convert_single(path, output)


def _convert_single(input_path: str, output_path: str = None):
    """Convert a single image to grayscale."""
    if output_path is None:
        output_path = input_path
    
    img = Image.open(input_path)
    gray_img = img.convert('L')
    gray_img.save(output_path)
    print(f"Converted: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images to grayscale')
    parser.add_argument('path', help='Path to image file or directory')
    parser.add_argument('--output', '-o', help='Output path (overwrites input if not specified)')
    parser.add_argument('--all', '-a', action='store_true', help='Convert all PNG files in directory')
    
    args = parser.parse_args()
    convert_to_grayscale(args.path, args.output, args.all)
