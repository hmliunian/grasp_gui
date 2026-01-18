import os
import base64
import io
import signal
import sys
import gc
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and current session
model = None
processor = None
device = None
current_image = None
current_inference_state = None
video_running = False
current_mask = None
current_overlay = None
place_mask = None

# -------------------------------
# Device setup
# -------------------------------
def setup_device():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

# -------------------------------
# Model setup
# -------------------------------
def setup_model():
    global model, processor
    print("Loading SAM3 model...")
    CHECKPOINT = "/home/xuran-yao/code/DISCOVERSE_v2_exp/sam3/checkpoints/sam3.pt"
    model = build_sam3_image_model(bpe_path=None,checkpoint_path=CHECKPOINT, enable_inst_interactivity=True)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)
    print("Model loaded successfully!")

def cleanup_resources():
    """Release GPU memory and other resources"""
    global model, processor, current_inference_state, current_image, device
    print("\nCleaning up resources...")
    
    # Clear inference state
    if current_inference_state is not None:
        # Try to clear any CUDA tensors in inference state
        try:
            if hasattr(current_inference_state, 'clear'):
                current_inference_state.clear()
        except:
            pass
        current_inference_state = None
    
    current_image = None
    
    # Move model to CPU and delete
    if model is not None:
        try:
            # Move model to CPU first to release GPU memory
            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                model = model.cpu()
                # Clear any remaining CUDA tensors
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error moving model to CPU: {e}")
        finally:
            del model
            model = None
    
    # Delete processor
    if processor is not None:
        del processor
        processor = None
    
    # Force garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()
    
    # Clear CUDA cache multiple times
    if torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU memory cleared. Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    print("Cleanup complete")
    sys.exit(0)

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    cleanup_resources()

# -------------------------------
# Helper functions
# -------------------------------
def convert_tensors_to_numpy(masks, scores, logits):
    """Convert tensors to numpy arrays to release GPU memory"""
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        masks = [m.cpu().numpy() if torch.is_tensor(m) else m for m in masks]
    
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()
    
    return masks, scores, logits

def cleanup_tensors(*tensors):
    """Delete tensors and clear CUDA cache"""
    for tensor in tensors:
        del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_mask_from_points(points_array, labels_array):
    """Predict mask from points and labels, return numpy arrays"""
    masks, scores, logits = model.predict_inst(
        current_inference_state,
        point_coords=points_array,
        point_labels=labels_array,
        multimask_output=False
    )
    return convert_tensors_to_numpy(masks, scores, logits)
def show_mask(mask, ax, color=[30/255,144/255,255/255,0.6]):
    h, w = mask.shape
    mask_img = np.zeros((h, w, 4))
    mask_img[..., :3] = color[:3]
    mask_img[..., 3] = mask * color[3]
    ax.imshow(mask_img)

def show_points(coords, labels, ax):
    pos = coords[labels==1]
    neg = coords[labels==0]
    ax.scatter(pos[:,0], pos[:,1], color='green', marker='*', s=200, edgecolor='white')
    ax.scatter(neg[:,0], neg[:,1], color='red', marker='*', s=200, edgecolor='white')

def create_mask_preview(image, masks, points=None, labels=None):
    """Create a base64 encoded preview image with masks and points"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for mask in masks:
        show_mask(mask, ax)

    if points is not None and labels is not None:
        show_points(points, labels, ax)

    ax.axis('off')

    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    # Encode to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# -------------------------------
# Flask routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_video', methods=['POST'])
def start_video():
    """Start video stream (snowflake video, can be replaced with camera)"""
    global video_running
    video_running = True
    
    # Generate a frame of snowflake-like video (random pixels with some structure)
    width, height = 640, 480
    # Create a dark background with random bright pixels (like snowflakes)
    frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    
    # Add random bright pixels (snowflakes)
    num_flakes = np.random.randint(100, 500)
    for _ in range(num_flakes):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(200, 256)
        frame[y, x] = [brightness, brightness, brightness]
    
    # Convert to base64
    img = Image.fromarray(frame)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'frame_data': f"data:image/jpeg;base64,{img_base64}",
        'width': width,
        'height': height
    })

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stop video and save last frame as current_image for SAM3"""
    global video_running, current_image, current_inference_state
    
    video_running = False
    
    # Get last frame from request (or generate one)
    data = request.get_json() or {}
    frame_data = data.get('frame_data')
    
    if frame_data:
        # Decode base64 frame
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        current_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    else:
        # Generate a random frame as fallback
        width, height = 640, 480
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        current_image = Image.fromarray(frame)
    
    # Clean up previous inference state
    if current_inference_state is not None:
        try:
            if hasattr(current_inference_state, 'clear'):
                current_inference_state.clear()
        except:
            pass
        del current_inference_state
        current_inference_state = None
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate inference state
    current_inference_state = processor.set_image(current_image)
    
    # Save current image
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'current_image.jpg')
    current_image.save(filename)
    
    # Convert to base64 for display
    buffered = io.BytesIO()
    current_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'image_data': f"data:image/jpeg;base64,{img_base64}",
        'image_path': filename,
        'image_size': {'width': current_image.width, 'height': current_image.height}
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    global current_image, current_inference_state

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
        # Clean up previous inference state to release GPU memory
        if current_inference_state is not None:
            try:
                if hasattr(current_inference_state, 'clear'):
                    current_inference_state.clear()
            except:
                pass
            del current_inference_state
            current_inference_state = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'current_image.jpg')
        file.save(filename)

        # Load and process image
        current_image = Image.open(filename).convert("RGB")

        # Generate inference state
        current_inference_state = processor.set_image(current_image)

        # Convert to base64 for display
        buffered = io.BytesIO()
        current_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'image_path': filename
        })

@app.route('/predict', methods=['POST'])
def predict_mask():
    """Legacy endpoint - kept for compatibility"""
    return predict_mask_new()

@app.route('/predict_mask', methods=['POST'])
def predict_mask_new():
    """Generate mask from points, return overlay and mask"""
    global current_image, current_inference_state, current_mask, current_overlay

    if current_image is None or current_inference_state is None:
        return jsonify({'error': 'No image loaded. Please stop video first.'}), 400

    data = request.get_json()
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not points or not labels:
        return jsonify({'error': 'No points provided'}), 400

    try:
        points_array = np.array(points)
        labels_array = np.array(labels)

        # Predict mask and convert to numpy
        masks, scores, logits = predict_mask_from_points(points_array, labels_array)

        # Save mask (single channel 0-255)
        current_mask = masks[0].astype(np.uint8) * 255

        # Create overlay (semi-transparent blue mask over original image)
        overlay = create_overlay_image(current_image, masks[0], points_array, labels_array)
        current_overlay = overlay

        # Convert overlay to base64
        buffered = io.BytesIO()
        overlay.save(buffered, format="PNG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Convert mask to base64
        mask_img = Image.fromarray(current_mask, mode='L')
        buffered_mask = io.BytesIO()
        mask_img.save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode('utf-8')

        # Save scores for response before cleanup
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores

        # Clean up temporary tensors
        cleanup_tensors(masks, scores, logits)

        return jsonify({
            'success': True,
            'overlay_data': f"data:image/png;base64,{overlay_base64}",
            'mask_data': f"data:image/png;base64,{mask_base64}",
            'scores': scores_list,
            'image_size': {'width': current_image.width, 'height': current_image.height}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_overlay_image(image, mask, points=None, labels=None):
    """Create overlay image with semi-transparent blue mask"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    # Show mask with semi-transparent blue
    show_mask(mask, ax, color=[30/255, 144/255, 255/255, 0.6])
    
    # Show points if provided
    if points is not None and labels is not None:
        show_points(points, labels, ax)
    
    ax.axis('off')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    # Load as PIL Image
    overlay_img = Image.open(buf).convert("RGB")
    return overlay_img

@app.route('/download_mask', methods=['POST'])
def download_mask():
    global current_image, current_inference_state

    if current_image is None or current_inference_state is None:
        return jsonify({'error': 'No image loaded'}), 400

    data = request.get_json()
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not points or not labels:
        return jsonify({'error': 'No points provided'}), 400

    try:
        points_array = np.array(points)
        labels_array = np.array(labels)

        # Predict mask and convert to numpy
        masks, scores, logits = predict_mask_from_points(points_array, labels_array)

        # Create mask image (grayscale, single channel)
        mask_array = masks[0].astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode='L')

        # Save to bytes buffer
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)

        # Clean up temporary tensors
        cleanup_tensors(masks, scores, logits)

        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='mask_output.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_combined', methods=['POST'])
def download_combined():
    global current_image, current_inference_state

    if current_image is None or current_inference_state is None:
        return jsonify({'error': 'No image loaded'}), 400

    data = request.get_json()
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not points or not labels:
        return jsonify({'error': 'No points provided'}), 400

    try:
        points_array = np.array(points)
        labels_array = np.array(labels)

        # Predict mask and convert to numpy
        masks, scores, logits = predict_mask_from_points(points_array, labels_array)

        # Create combined image with mask overlay
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(current_image)

        for mask in masks:
            show_mask(mask, ax)

        if points_array is not None and labels_array is not None:
            show_points(points_array, labels_array, ax)

        ax.axis('off')

        # Save to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)

        # Clean up temporary tensors
        cleanup_tensors(masks, scores, logits)

        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='combined_output.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_place', methods=['POST'])
def select_place():
    """Select 10x10 placement point region, return full-size mask"""
    global current_image, place_mask

    if current_image is None:
        return jsonify({'error': 'No image available. Please stop video first.'}), 400

    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({'error': 'Placement coordinates not provided'}), 400

    try:
        h, w = current_image.height, current_image.width
        
        # Ensure coordinates are within valid range
        x = max(0, min(int(x), w - 10))
        y = max(0, min(int(y), h - 10))

        # Create full-size black background mask (single channel grayscale)
        # Black background (0) + white selected region (255)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y:y+10, x:x+10] = 255
        place_mask = full_mask

        # Convert to base64 for preview
        mask_image = Image.fromarray(full_mask, mode='L')
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)
        mask_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'mask_data': f"data:image/png;base64,{mask_base64}",
            'coordinates': {'x': x, 'y': y},
            'image_size': {'width': w, 'height': h}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_region_mask', methods=['POST'])
def download_region_mask():
    """Legacy endpoint - kept for compatibility"""
    return download_place_mask()

@app.route('/download', methods=['POST'])
def download():
    """Universal download endpoint for overlay, mask, or place mask"""
    global current_mask, current_overlay, place_mask

    data = request.get_json()
    download_type = data.get('type')  # 'overlay', 'mask', or 'place_mask'

    if download_type == 'overlay':
        if current_overlay is None:
            return jsonify({'error': 'No overlay available. Please generate mask first.'}), 400
        buf = io.BytesIO()
        current_overlay.save(buf, format='PNG')
        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='overlay_output.png'
        )
    
    elif download_type == 'mask':
        if current_mask is None:
            return jsonify({'error': 'No mask available. Please generate mask first.'}), 400
        mask_image = Image.fromarray(current_mask, mode='L')
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='mask_output.png'
        )
    
    elif download_type == 'place_mask':
        if place_mask is None:
            return jsonify({'error': 'No placement mask available. Please select placement point first.'}), 400
        mask_image = Image.fromarray(place_mask, mode='L')
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='place_mask.png'
        )
    
    else:
        return jsonify({'error': 'Invalid download type. Use "overlay", "mask", or "place_mask".'}), 400

def download_place_mask():
    """Download placement mask"""
    global place_mask

    if place_mask is None:
        return jsonify({'error': 'No placement mask available. Please select placement point first.'}), 400

    mask_image = Image.fromarray(place_mask, mode='L')
    buf = io.BytesIO()
    mask_image.save(buf, format='PNG')
    buf.seek(0)

    return send_file(
        buf,
        mimetype='image/png',
        as_attachment=True,
        download_name='place_mask.png'
    )

if __name__ == '__main__':
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    setup_device()
    setup_model()
    app.run(host='0.0.0.0', port=50052, debug=True)
