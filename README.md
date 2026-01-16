# SAM3 Web GUI

A web-based graphical user interface for SAM3 (Segment Anything Model 3) image segmentation.

## Features

- **Image Upload**: Drag and drop or click to upload images
- **Interactive Point Selection**: Add positive (foreground) and negative (background) points by clicking
- **Real-time Preview**: Generate and preview segmentation masks
- **Save Results**: Download both the mask and combined overlay image
- **Placement Point Selection**: Select a 10x10 pixel region on the mask preview as placement point (independent of SAM3 mask)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the SAM3 package installed and available.

## Usage

1. Run the Flask application:
```bash
python sam_gui.py
```

2. Open your web browser and navigate to `http://localhost:50052`

3. Upload an image using the upload area

4. Select point mode:
   - **Positive Point (Green)**: Click to mark foreground regions
   - **Negative Point (Red)**: Click to mark background regions

5. Click on the image to add points. Click on existing points to remove them.

6. Click "Generate Mask" to see the segmentation result

7. Click "Save Result" to download the mask and combined image

8. (Optional) Click on the mask preview to select a 10x10 pixel region as placement point, then click "Download Region Mask" to save it

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload an image file
- `POST /predict`: Generate mask prediction from points
- `POST /save`: Save mask and combined results

## File Structure

```
sam_gui/
├── sam_gui.py          # Main Flask application
├── templates/
│   └── index.html      # Web interface template
├── static/
│   └── uploads/        # Uploaded images and results
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.7+
- Flask
- PyTorch
- PIL (Pillow)
- NumPy
- Matplotlib
- SAM3 package
