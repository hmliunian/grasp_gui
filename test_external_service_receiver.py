#!/usr/bin/env python3
"""
External Service Receiver - Test Process 1

This Flask server simulates an external service that receives images from the GUI.
It listens for POST requests at /api/process_image and saves received images to disk.

Usage:
    python test_external_service_receiver.py
    
Make sure to update EXTERNAL_SERVICE_URL in sam_gui.py to point to this server:
    EXTERNAL_SERVICE_URL = "http://localhost:50053/api/process_image"
"""

import os
import base64
import io
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Configuration
RECEIVER_PORT = 50053
OUTPUT_DIR = "test_outputs/received_images"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/api/process_image', methods=['POST'])
def receive_image():
    """Receive image from GUI's /external/send_image endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract image data
        image_data = data.get('image')
        width = data.get('width')
        height = data.get('height')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"received_image_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save image
        img.save(filepath, format='PNG')
        
        # Log information
        actual_width, actual_height = img.size
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Image received")
        print(f"  Request width: {width}, height: {height}")
        print(f"  Actual size: {actual_width}x{actual_height}")
        print(f"  Saved to: {filepath}")
        print(f"  File size: {file_size:.2f} KB")
        print("-" * 60)
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Image received and saved successfully',
            'filename': filename,
            'filepath': filepath,
            'width': actual_width,
            'height': actual_height
        }), 200
        
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        print(f"ERROR: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'External Service Receiver',
        'output_dir': OUTPUT_DIR
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("External Service Receiver")
    print("=" * 60)
    print(f"Listening on: http://localhost:{RECEIVER_PORT}/api/process_image")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Health check: http://localhost:{RECEIVER_PORT}/health")
    print("=" * 60)
    print("Make sure to update EXTERNAL_SERVICE_URL in sam_gui.py:")
    print(f'  EXTERNAL_SERVICE_URL = "http://localhost:{RECEIVER_PORT}/api/process_image"')
    print("=" * 60)
    print()
    
    app.run(host='0.0.0.0', port=RECEIVER_PORT, debug=True)
