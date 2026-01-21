import os
import io
import sys
import uuid
import time
import signal
import base64
import logging
import threading
import asyncio
import numpy as np
import cv2
import torch
import zenoh
import requests
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from generated.image_pb2 import ImageData
from generated.action_pb2 import GraspGoal

# --- Configuration ---
CONFIG = {
    'CHECKPOINT_PATH': "compute/third_party/sam3/ckp/sam3.pt",
    'DEVICE': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'CAMERA_ID': "SN_57524755", 
    'ZENOH_ROUTER': "tcp/10.42.0.220:7447#so_sndbuf=52428800",
    'ACTION_GRASP_GOAL': "arm/action/grasp/goal",
    'EXTERNAL_SERVICE_URL': os.getenv("EXTERNAL_SERVICE_URL", "http://noelia-pupillary-uneccentrically.ngrok-free.dev/api/process_image")
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class PointsData(BaseModel):
    points: List[List[float]]
    labels: List[int]

class ExternalMaskData(BaseModel):
    mask_data: str

class PlaceData(BaseModel):
    x: float
    y: float

# --- Helper Classes ---
class ImageProcessor:
    @staticmethod
    def to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=quality, optimize=True)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        mime = "jpeg" if fmt.upper() in ["JPG", "JPEG"] else "png"
        return f"data:image/{mime};base64,{img_str}"

    @staticmethod
    def from_base64(base64_str: str) -> Image.Image:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")

    @staticmethod
    def create_overlay(image: Image.Image, mask_uint8: np.ndarray, points=None, labels=None) -> Image.Image:
        # Note: mask_uint8 must be (H, W) and match image dimensions
        overlay_rgba = Image.new("RGBA", image.size, (30, 144, 255, 0))
        overlay_np = np.array(overlay_rgba)
        
        # Alpha channel: 150 where mask is present
        overlay_np[..., 3] = np.where(mask_uint8 > 0, 150, 0).astype(np.uint8)
        
        overlay_layer = Image.fromarray(overlay_np, "RGBA")
        result = Image.alpha_composite(image.convert("RGBA"), overlay_layer).convert("RGB")

        if points is not None and len(points) > 0:
            draw = ImageDraw.Draw(result)
            r = 6
            for point, label in zip(points, labels):
                x, y = point
                color = "#10b981" if label == 1 else "#ef4444"
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="white", width=2)
                
        return result

    @staticmethod
    def normalize_mask(mask_raw: np.ndarray, target_pil_size: tuple) -> np.ndarray:
        """
        Robustly ensures mask is 2D (H, W), binary uint8 (0 or 255), 
        and resized to match the target PIL size (W, H).
        """
        # 1. Ensure 2D (H, W) by squeezing extra dimensions
        mask = np.squeeze(mask_raw)
        if mask.ndim > 2:
             # Safety fallback: if squeeze still leaves 3D (e.g. C,H,W), take first channel
             logger.warning(f"Mask raw shape {mask_raw.shape} resulted in {mask.shape} after squeeze. Taking first channel.")
             mask = mask[0, :, :]

        # 2. Convert to strictly binary uint8 (0 or 255)
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        elif np.issubdtype(mask.dtype, np.floating):
            # Threshold floats. If logits (neg/pos), threshold 0. If prob (0-1), threshold 0.5.
            threshold = 0.0 if (mask.min() < 0 or mask.max() > 1.0) else 0.5
            mask_uint8 = (mask > threshold).astype(np.uint8) * 255
        else:
            # Handle integer input
            mask_uint8 = mask.astype(np.uint8)
            # If it's 0/1, convert to 0/255
            if mask_uint8.max() <= 1:
                 mask_uint8 *= 255
            # Ensure strictly binary 0 or 255
            mask_uint8 = (mask_uint8 > 127).astype(np.uint8) * 255

        # 3. Resize to match target dimensions if necessary
        target_w, target_h = target_pil_size
        current_h, current_w = mask_uint8.shape

        if (current_w, current_h) != (target_w, target_h):
             logger.info(f"Resizing mask from ({current_w}, {current_h}) to ({target_w}, {target_h})")
             # Use PIL for resizing binary masks (NEAREST neighbor is crucial)
             mask_pil = Image.fromarray(mask_uint8)
             mask_pil = mask_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
             mask_uint8 = np.array(mask_pil)
             
        return mask_uint8

class ImageDecoders:
    @staticmethod
    def decode_rgb(msg: ImageData) -> Optional[np.ndarray]:
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

# --- Zenoh Logic ---
class ZenohStreamer:
    def __init__(self):
        self.session = None
        self.subscriber_rgb = None
        self.subscriber_pc = None
        self.publisher_mask = None
        self.publisher_action = None
        self._latest_image = None
        self._latest_mask_pc = None
        self._lock = threading.Lock()
        self._is_streaming = False
        self._init_session()

    def _init_session(self):
        try:
            conf = zenoh.Config()
            conf.insert_json5("scouting/multicast/enabled", "true")
            conf.insert_json5("transport/shared_memory/enabled", "false")
            if CONFIG['ZENOH_ROUTER']:
                conf.insert_json5("connect/endpoints", f"['{CONFIG['ZENOH_ROUTER']}']")
            logger.info("Initializing Zenoh Session...")
            self.session = zenoh.open(conf)
        except Exception as e:
            logger.error(f"Failed to initialize Zenoh session: {e}")

    def start_stream(self):
        with self._lock:
            if self._is_streaming: return True
            if not self.session: self._init_session()
            camera_id = CONFIG['CAMERA_ID']
            rgb_topic = f"env/{camera_id}/rgb"
            mask_topic = f"env/{camera_id}/mask"
            pc_topic = f"env/{camera_id}/mask_pc"
            try:
                self.subscriber_rgb = self.session.declare_subscriber(rgb_topic, self._on_image_update)
                self.subscriber_pc = self.session.declare_subscriber(pc_topic, self._on_pc_update)
                self.publisher_mask = self.session.declare_publisher(mask_topic)
                self.publisher_action = self.session.declare_publisher(CONFIG['ACTION_GRASP_GOAL'])
                self._is_streaming = True
                logger.info("Zenoh Stream Started")
                return True
            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                return False

    def stop_stream(self):
        with self._lock:
            if not self._is_streaming: return
            if self.subscriber_rgb: self.subscriber_rgb.undeclare(); self.subscriber_rgb = None
            if self.subscriber_pc: self.subscriber_pc.undeclare(); self.subscriber_pc = None
            if self.publisher_mask: self.publisher_mask.undeclare(); self.publisher_mask = None
            if self.publisher_action: self.publisher_action.undeclare(); self.publisher_action = None
            self._latest_image = None
            self._is_streaming = False
            logger.info("Zenoh Stream Stopped")

    def _on_image_update(self, sample):
        try:
            msg = ImageData()
            msg.ParseFromString(bytes(sample.payload))
            fmt = getattr(msg, 'format', '')
            if fmt in ["jpeg", "jpg", "png"] or not fmt:
                np_img = ImageDecoders.decode_rgb(msg)
                if np_img is not None:
                    pil_img = Image.fromarray(np_img)
                    with self._lock:
                        self._latest_image = pil_img
        except Exception:
            pass

    def _on_pc_update(self, sample):
        with self._lock:
            self._latest_mask_pc = sample.payload
            logger.info(f"Received Mask PC update. Bytes: {len(self._latest_mask_pc)}")

    def publish_mask(self, mask_arr: np.ndarray):
        if not self.publisher_mask and self.session: self.start_stream()
        if not self.publisher_mask: return
        try:
            # mask_arr must be (H, W) uint8
            success, encoded_mask = cv2.imencode(".png", mask_arr)
            if success:
                out = ImageData()
                out.height = mask_arr.shape[0]
                out.width = mask_arr.shape[1]
                out.channels = 1
                out.format = "png"
                out.data = encoded_mask.tobytes()
                with self._lock: self._latest_mask_pc = None 
                self.publisher_mask.put(out.SerializeToString())
                logger.info(f"Mask published (Shape: {mask_arr.shape})")
        except Exception as e:
            logger.error(f"Failed to publish mask: {e}")

    async def wait_and_publish_action(self, timeout=5.0):
        start_time = time.time()
        logger.info("Waiting for Mask Point Cloud from Environment...")
        pc_data = None
        while time.time() - start_time < timeout:
            with self._lock:
                if self._latest_mask_pc is not None:
                    pc_data = self._latest_mask_pc
                    break
            await asyncio.sleep(0.1)
        
        if pc_data is None: return False, "Timeout waiting for Mask PC"

        try:
            action_id = str(uuid.uuid4())
            goal = GraspGoal()
            goal.action_id = action_id
            goal.mask_pc = bytes(pc_data)
            goal.initial_approach_pose.position.x = 0.3
            goal.initial_approach_pose.position.y = 0.0
            goal.initial_approach_pose.position.z = 0.28
            goal.initial_approach_pose.quaternion.x = 0.0
            goal.initial_approach_pose.quaternion.y = 0.229
            goal.initial_approach_pose.quaternion.z = 0
            goal.initial_approach_pose.quaternion.w = 0.973
            goal.retract_pose.CopyFrom(goal.initial_approach_pose)

            if not self.publisher_action: self.start_stream()
            self.publisher_action.put(goal.SerializeToString())
            logger.info(f"GraspAction Sent! ID: {action_id}")
            return True, action_id
        except Exception as e:
            return False, str(e)

    def get_latest_frame(self):
        with self._lock:
            return self._latest_image.copy() if self._latest_image else None
    
    def is_active(self): return self._is_streaming
    def close(self):
        self.stop_stream()
        if self.session: self.session.close()

# --- SAM3 & Session Logic ---
class SAM3ModelHandler:
    def __init__(self):
        self.device = CONFIG['DEVICE']
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading SAM3 on {self.device}...")
        if not os.path.exists(CONFIG['CHECKPOINT_PATH']):
            raise FileNotFoundError(f"Checkpoint not found at {CONFIG['CHECKPOINT_PATH']}")
        try:
            self.model = build_sam3_image_model(
                checkpoint_path=CONFIG['CHECKPOINT_PATH'], 
                device=self.device,
                enable_inst_interactivity=True
            )
            self.model.eval()
            self.processor = Sam3Processor(self.model)
            logger.info("SAM3 Model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise e

    def get_inference_state(self, image: Image.Image):
        if self.device == 'cuda': torch.cuda.empty_cache()
        return self.processor.set_image(image)

    @torch.inference_mode()
    def predict(self, inference_state, points, labels):
        masks, scores, logits = self.model.predict_inst(
            inference_state,
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        return self._to_numpy(masks), self._to_numpy(scores)

    def _to_numpy(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.float().cpu().numpy()
        if isinstance(tensor, list):
            return [t.float().cpu().numpy() if torch.is_tensor(t) else t for t in tensor]
        return tensor

class SessionManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_image = None
        self.release_inference()
        self.current_mask = None
        self.current_overlay = None
        self.external_mask_buffer = None
        self.external_mask_locked = False
        self.mask_source_mode = None 

    def set_image(self, image: Image.Image, inference_state):
        self.release_inference() 
        self.current_image = image
        self.inference_state = inference_state
        self.current_mask = None
        self.current_overlay = None
        self.external_mask_buffer = None
        self.external_mask_locked = False
        self.mask_source_mode = None

    def release_inference(self):
        if hasattr(self, 'inference_state') and self.inference_state and hasattr(self.inference_state, 'clear'):
            try: self.inference_state.clear()
            except: pass
        self.inference_state = None

# --- FastAPI App ---
app = FastAPI(title="SAM3 Robot Teleop")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="compute/third_party/grasp_gui/templates")

model_handler = SAM3ModelHandler()
session_manager = SessionManager()
zenoh_streamer = ZenohStreamer()

zenoh_streamer.start_stream()

def _capture_and_encode_latest_frame():
    latest_img = zenoh_streamer.get_latest_frame()
    if latest_img is None:
        raise ValueError("No video stream frame available yet.")
    logger.info(f"Encoding new frame (Full Res: {latest_img.size})...")
    inf_state = model_handler.get_inference_state(latest_img)
    session_manager.set_image(latest_img, inf_state)

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get_frame")
def get_frame():
    if session_manager.current_overlay:
         return {
            "success": True, 
            "status": "overlay", 
            "frame_data": ImageProcessor.to_base64(session_manager.current_overlay, "PNG")
        }

    img = zenoh_streamer.get_latest_frame()
    if not img:
        return {"success": True, "status": "waiting"}
    
    return {
        "success": True, 
        "status": "streaming", 
        "frame_data": ImageProcessor.to_base64(img, "JPEG", quality=80)
    }

@app.post("/reset")
async def reset_session():
    session_manager.reset()
    return {"success": True}

@app.post("/predict_mask")
def predict_mask(data: PointsData):
    if not data.points:
        return JSONResponse(status_code=400, content={"error": "No points provided"})
    try:
        # 1. Capture Full Res
        _capture_and_encode_latest_frame()
        points_arr = np.array(data.points)
        labels_arr = np.array(data.labels)
        
        # 2. SAM Prediction
        masks, scores = model_handler.predict(session_manager.inference_state, points_arr, labels_arr)
        
        # 3. Normalize Mask (Critical Fix: Resize to match image dimensions)
        target_size = session_manager.current_image.size # (W, H)
        final_mask = ImageProcessor.normalize_mask(masks[0], target_size)
        
        session_manager.current_mask = final_mask
        session_manager.mask_source_mode = 'sam3'
        
        # 4. Publish correct size mask
        zenoh_streamer.publish_mask(session_manager.current_mask)
        
        # 5. Create Overlay
        overlay_img = ImageProcessor.create_overlay(
            session_manager.current_image, final_mask, points_arr, labels_arr
        )
        session_manager.current_overlay = overlay_img

        return {"success": True}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/external/send_image")
async def external_send_image():
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _capture_and_encode_latest_frame)
        img_base64 = ImageProcessor.to_base64(session_manager.current_image, "PNG")
        payload = {
            'image': img_base64,
            'width': session_manager.current_image.width,
            'height': session_manager.current_image.height
        }
        session_manager.external_mask_buffer = None
        session_manager.external_mask_locked = False
        session_manager.mask_source_mode = 'external'
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(CONFIG['EXTERNAL_SERVICE_URL'], json=payload, timeout=10)
        )
        response.raise_for_status()
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed: {str(e)}"})

@app.get("/external/health")
async def external_health():
    target_url = CONFIG['EXTERNAL_SERVICE_URL']
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, requests.get, target_url, {'timeout': 2})
        return {"status": "online", "url": target_url}
    except Exception:
        return {"status": "offline", "url": target_url}

@app.post("/external/receive_mask")
async def external_receive_mask_post(data: ExternalMaskData):
    if session_manager.current_image is None:
        return JSONResponse(status_code=400, content={"error": "No image loaded"})
    if session_manager.external_mask_locked:
        return {"success": False, "message": "Mask already confirmed."}
    try:
        mask_data = data.mask_data
        mask_img = ImageProcessor.from_base64(mask_data).convert('L')
        mask_array = np.array(mask_img)
        
        # Normalize Mask (Resize to match current image)
        target_size = session_manager.current_image.size
        final_mask = ImageProcessor.normalize_mask(mask_array, target_size)

        session_manager.external_mask_buffer = final_mask
        return {"success": True, "message": "Mask stored in buffer"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process mask: {str(e)}"})

@app.get("/external/receive_mask")
def external_receive_mask_get(confirm: bool = False):
    if session_manager.current_image is None:
        return JSONResponse(status_code=400, content={"error": "No image captured yet."})
    if session_manager.external_mask_buffer is None:
        return JSONResponse(status_code=400, content={"error": "No mask in buffer."})
    try:
        mask_buffer = session_manager.external_mask_buffer
        overlay_img = ImageProcessor.create_overlay(session_manager.current_image, mask_buffer)
        if confirm:
            session_manager.current_mask = mask_buffer.copy()
            session_manager.current_overlay = overlay_img
            session_manager.external_mask_locked = True
            session_manager.mask_source_mode = 'external'
            zenoh_streamer.publish_mask(session_manager.current_mask)
            return {"success": True, "locked": True}
        else:
            session_manager.current_overlay = overlay_img
            return {"success": True, "preview": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/execute_grasp")
async def execute_grasp():
    if session_manager.current_mask is None:
        return JSONResponse(status_code=400, content={"error": "No mask generated yet."})
    success, result = await zenoh_streamer.wait_and_publish_action(timeout=5.0)
    if success:
        # Don't clear overlay, allow re-execution
        return {"success": True, "action_id": result, "message": "Grasp command sent!"}
    else:
        return JSONResponse(status_code=504, content={"error": result})

@app.post("/select_place")
async def select_place(data: PlaceData):
    return {"success": True}

def signal_handler(sig, frame):
    zenoh_streamer.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50052, workers=1)