import os
import base64
import io
import json
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

# -------------------------------
# Helper functions
# -------------------------------
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

@app.route('/upload', methods=['POST'])
def upload_image():
    global current_image, current_inference_state

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
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
    global current_image, current_inference_state

    if current_image is None or current_inference_state is None:
        return jsonify({'error': 'No image loaded'}), 400

    data = request.get_json()
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not points or not labels:
        return jsonify({'error': 'No points provided'}), 400

    try:
        # Convert to numpy arrays
        points_array = np.array(points)
        labels_array = np.array(labels)

        # Predict mask
        masks, scores, logits = model.predict_inst(
            current_inference_state,
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False
        )

        # Create preview
        preview_data = create_mask_preview(current_image, masks, points_array, labels_array)

        return jsonify({
            'success': True,
            'mask_preview': preview_data,
            'scores': scores.tolist() if hasattr(scores, 'tolist') else scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        # Generate mask
        points_array = np.array(points)
        labels_array = np.array(labels)

        masks, scores, logits = model.predict_inst(
            current_inference_state,
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False
        )

        # Create mask image in memory
        mask_array = masks[0].astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array)

        # Save to bytes buffer
        buf = io.BytesIO()
        mask_image.save(buf, format='PNG')
        buf.seek(0)

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
        # Generate mask
        points_array = np.array(points)
        labels_array = np.array(labels)

        masks, scores, logits = model.predict_inst(
            current_inference_state,
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False
        )

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

        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name='combined_output.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_device()
    setup_model()
    app.run(host='0.0.0.0', port=50052, debug=True)
