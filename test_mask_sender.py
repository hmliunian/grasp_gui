#!/usr/bin/env python3
"""
Mask Sender - Test Process 2

This script reads mask files from disk and sends them to the GUI's
/external/receive_mask endpoint via POST request.

Usage:
    # Send a single mask file
    python test_mask_sender.py --mask_path path/to/mask.png
    
    # Send all masks from a directory
    python test_mask_sender.py --mask_dir test_outputs/masks_to_send
    
    # Send with specific GUI URL (default: http://localhost:50052)
    python test_mask_sender.py --mask_path mask.png --gui_url http://localhost:50052
"""

import os
import sys
import argparse
import base64
import io
from pathlib import Path
from PIL import Image
import requests

# Default configuration
DEFAULT_GUI_URL = "http://localhost:50052"
DEFAULT_MASK_DIR = "test_outputs/masks_to_send"

def mask_to_base64(mask_path):
    """Convert mask image file to base64 data URI"""
    try:
        img = Image.open(mask_path).convert('L')  # Ensure grayscale
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{mask_base64}"
    except Exception as e:
        raise Exception(f"Failed to process mask file {mask_path}: {str(e)}")

def send_mask(mask_path, gui_url):
    """Send a single mask file to the GUI"""
    print(f"Sending mask: {mask_path}")
    
    try:
        # Convert mask to base64
        mask_data = mask_to_base64(mask_path)
        
        # Prepare payload
        payload = {'mask_data': mask_data}
        
        # Send POST request
        endpoint = f"{gui_url}/external/receive_mask"
        print(f"  Target: {endpoint}")
        
        response = requests.post(
            endpoint,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"  ✓ Mask sent successfully")
                print(f"  Message: {data.get('message', 'N/A')}")
                return True
            else:
                print(f"  ✗ Request failed: {data.get('error', data.get('message', 'Unknown error'))}")
                return False
        else:
            print(f"  ✗ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection error: Could not connect to {gui_url}")
        print(f"    Make sure the GUI server (sam_gui.py) is running")
        return False
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timeout")
        return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def send_masks_from_directory(mask_dir, gui_url):
    """Send all mask files from a directory"""
    mask_dir = Path(mask_dir)
    
    if not mask_dir.exists():
        print(f"Error: Directory does not exist: {mask_dir}")
        return False
    
    if not mask_dir.is_dir():
        print(f"Error: Not a directory: {mask_dir}")
        return False
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    mask_files = [f for f in mask_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not mask_files:
        print(f"No image files found in: {mask_dir}")
        return False
    
    print(f"Found {len(mask_files)} mask file(s) in {mask_dir}")
    print("=" * 60)
    
    success_count = 0
    for mask_file in sorted(mask_files):
        if send_mask(mask_file, gui_url):
            success_count += 1
        print()  # Blank line between files
    
    print("=" * 60)
    print(f"Summary: {success_count}/{len(mask_files)} masks sent successfully")
    print("=" * 60)
    
    return success_count == len(mask_files)

def main():
    parser = argparse.ArgumentParser(
        description='Send mask files to GUI /external/receive_mask endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send a single mask file
  python test_mask_sender.py --mask_path mask.png
  
  # Send all masks from a directory
  python test_mask_sender.py --mask_dir test_outputs/masks_to_send
  
  # Use custom GUI URL
  python test_mask_sender.py --mask_path mask.png --gui_url http://localhost:50052
        """
    )
    
    parser.add_argument(
        '--mask_path',
        type=str,
        help='Path to a single mask image file to send'
    )
    
    parser.add_argument(
        '--mask_dir',
        type=str,
        default=None,
        help=f'Directory containing mask files to send (default: {DEFAULT_MASK_DIR})'
    )
    
    parser.add_argument(
        '--gui_url',
        type=str,
        default=DEFAULT_GUI_URL,
        help=f'GUI server URL (default: {DEFAULT_GUI_URL})'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mask_path and not args.mask_dir:
        # Try default directory if nothing specified
        if os.path.exists(DEFAULT_MASK_DIR):
            print(f"No arguments provided, using default directory: {DEFAULT_MASK_DIR}")
            args.mask_dir = DEFAULT_MASK_DIR
        else:
            parser.print_help()
            print(f"\nError: Either --mask_path or --mask_dir must be provided")
            print(f"       (or ensure default directory exists: {DEFAULT_MASK_DIR})")
            sys.exit(1)
    
    if args.mask_path and args.mask_dir:
        parser.print_help()
        print("\nError: Cannot specify both --mask_path and --mask_dir")
        sys.exit(1)
    
    print("=" * 60)
    print("Mask Sender - Test Process 2")
    print("=" * 60)
    print(f"GUI URL: {args.gui_url}")
    print("=" * 60)
    print()
    
    # Send mask(s)
    if args.mask_path:
        # Single file
        if not os.path.exists(args.mask_path):
            print(f"Error: Mask file does not exist: {args.mask_path}")
            sys.exit(1)
        
        success = send_mask(args.mask_path, args.gui_url)
        sys.exit(0 if success else 1)
    
    else:
        # Directory
        mask_dir = args.mask_dir or DEFAULT_MASK_DIR
        success = send_masks_from_directory(mask_dir, args.gui_url)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
