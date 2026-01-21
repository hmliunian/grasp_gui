import os
import io
import sys
import gc
import uuid
import time
import signal
import base64
import logging
import threading
import numpy as np
import cv2
import torch
import zenoh
from functools import partial
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_file, render_template

# --- SAM3 Imports ---
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Protobuf Imports (Mocked for this context) ---
# In production, ensure these are compiled from your .proto files
try:
    from generated.image_pb2 import ImageData
    from generated.action_pb2 import GraspGoal
    from generated.types_pb2 import Pose, Vector3, Quaternion
except ImportError:
    print("Warning: Generated protos not found. Creating Mocks.")
    class MockProto:
        def SerializeToString(self): return b'mock_bytes'
        def ParseFromString(self, data): pass
        def __init__(self, **kwargs): 
            for k,v in kwargs.items(): setattr(self, k, v)
    
    class ImageData(MockProto):
        class Timestamp:
            def GetCurrentTime(self): pass
        def __init__(self): self.timestamp = self.Timestamp()
    
    class GraspGoal(MockProto): pass
    class Pose(MockProto): pass

# --- Configuration ---
CONFIG = {
    'UPLOAD_FOLDER': 'static/uploads',
    'CHECKPOINT_PATH': "compute/third_party/sam3/ckp/sam3.pt",
    'MAX_IMAGE_SIZE': 1024,
    'DEVICE': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'DTYPE': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    
    # Zenoh Configuration
    'CAMERA_ID': "SN_57524755", 
    'ZENOH_ROUTER': "tcp/10.42.0.220:7447#so_sndbuf=52428800",
    
    # Action Topics
    'ACTION_GRASP_GOAL': "arm/action/grasp/goal"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Classes ---
class ImageProcessor:
    @staticmethod
    def resize_if_needed(image: Image.Image, max_size: int) -> Image.Image:
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

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
    def create_overlay(image: Image.Image, mask_array: np.ndarray, points=None, labels=None) -> Image.Image:
        mask_uint8 = (mask_array > 0).astype(np.uint8) * 255
        overlay_rgba = Image.new("RGBA", image.size, (30, 144, 255, 0))
        overlay_np = np.array(overlay_rgba)
        
        if mask_uint8.shape != (image.height, image.width):
             mask_img = Image.fromarray(mask_uint8).resize(image.size, Image.Resampling.NEAREST)
             mask_uint8 = np.array(mask_img)

        overlay_np[..., 3] = np.where(mask_uint8 > 0, 150, 0).astype(np.uint8)
        overlay_layer = Image.fromarray(overlay_np, "RGBA")
        result = Image.alpha_composite(image.convert("RGBA"), overlay_layer).convert("RGB")

        if points is not None and len(points) > 0:
            draw = ImageDraw.Draw(result)
            r = 5
            for point, label in zip(points, labels):
                x, y = point
                color = "green" if label == 1 else "red"
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="white", width=2)
        return result

class ImageDecoders:
    @staticmethod
    def decode_rgb(msg: ImageData) -> np.ndarray:
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
        self.subscriber_pc = None  # New: Listen for Point Cloud
        self.publisher_mask = None
        self.publisher_action = None # New: Publish Action
        
        self._latest_image = None
        self._latest_mask_pc = None # Stores the incoming point cloud
        self._mask_pc_timestamp = 0
        
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
            
            # --- FIX: Construct Topics correctly from CAMERA_ID ---
            rgb_topic = f"env/{camera_id}/rgb"
            mask_topic = f"env/{camera_id}/mask"
            pc_topic = f"env/{camera_id}/mask_pc"
            # -----------------------------------------------------

            logger.info(f"Subscribing to RGB: {rgb_topic}")
            logger.info(f"Subscribing to PC: {pc_topic}")
            logger.info(f"Publisher Mask: {mask_topic}")
            logger.info(f"Publisher Action: {CONFIG['ACTION_GRASP_GOAL']}")

            try:
                # 1. RGB Sub
                self.subscriber_rgb = self.session.declare_subscriber(
                    rgb_topic, self._on_image_update
                )
                
                # 2. Mask PC Sub (Wait for env to process mask)
                self.subscriber_pc = self.session.declare_subscriber(
                    pc_topic, self._on_pc_update
                )

                # 3. Publishers
                self.publisher_mask = self.session.declare_publisher(mask_topic)
                self.publisher_action = self.session.declare_publisher(CONFIG['ACTION_GRASP_GOAL'])

                self._is_streaming = True
                return True
            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                return False

    def stop_stream(self):
        with self._lock:
            if not self._is_streaming: return
            if self.subscriber_rgb:
                self.subscriber_rgb.undeclare()
                self.subscriber_rgb = None
            if self.subscriber_pc:
                self.subscriber_pc.undeclare()
                self.subscriber_pc = None
            if self.publisher_mask:
                self.publisher_mask.undeclare()
                self.publisher_mask = None
            if self.publisher_action:
                self.publisher_action.undeclare()
                self.publisher_action = None

            self._latest_image = None
            self._is_streaming = False

    # --- Callbacks ---

    def _on_image_update(self, sample):
        try:
            msg = ImageData()
            msg.ParseFromString(bytes(sample.payload))
            if getattr(msg, 'format', '') in ["jpeg", "jpg", "png"]:
                np_img = ImageDecoders.decode_rgb(msg)
                if np_img is not None:
                    pil_img = Image.fromarray(np_img)
                    with self._lock:
                        self._latest_image = pil_img
        except Exception:
            pass

    def _on_pc_update(self, sample):
        """Called when env returns the processed point cloud from the mask."""
        with self._lock:
            # We store the raw bytes because we just pass them to the Action
            self._latest_mask_pc = sample.payload
            self._mask_pc_timestamp = time.time()
            logger.info(f"Received Mask PC update. Bytes: {len(self._latest_mask_pc)}")

    # --- Actions ---

    def publish_mask(self, mask_arr: np.ndarray):
        """Publishes mask to env, which should trigger a PC calculation."""
        if not self.publisher_mask and self.session:
             self.start_stream() # Ensure pubs exist

        if not self.publisher_mask:
            logger.error("Cannot publish mask: Publisher not active.")
            return

        try:
            success, encoded_mask = cv2.imencode(".png", mask_arr)
            if success:
                out = ImageData()
                if hasattr(out, 'timestamp'):
                    try: out.timestamp.GetCurrentTime()
                    except: pass
                
                out.height = mask_arr.shape[0]
                out.width = mask_arr.shape[1]
                out.channels = 1
                out.format = "png"
                out.data = encoded_mask.tobytes()
                
                # Clear previous PC so we know when the new one arrives
                with self._lock:
                    self._latest_mask_pc = None 
                
                self.publisher_mask.put(out.SerializeToString())
                logger.info(f"Mask published to Zenoh.")
        except Exception as e:
            logger.error(f"Failed to publish mask: {e}")

    def wait_and_publish_action(self, timeout=5.0):
        """Waits for mask_pc to arrive, then sends GraspGoal."""
        start_time = time.time()
        logger.info("Waiting for Mask Point Cloud from Environment...")
        
        pc_data = None
        while time.time() - start_time < timeout:
            with self._lock:
                if self._latest_mask_pc is not None:
                    pc_data = self._latest_mask_pc
                    break
            time.sleep(0.1)
        
        if pc_data is None:
            logger.error("Timeout: Mask PC was not received from environment.")
            return False, "Timeout waiting for Mask PC"

        # Construct Action Goal
        try:
            action_id = str(uuid.uuid4())
            goal = GraspGoal()
            goal.action_id = action_id
            goal.mask_pc = bytes(pc_data) # The raw PC bytes
            goal.initial_approach_pose.position.x = 0.3
            goal.initial_approach_pose.position.y = 0.0
            goal.initial_approach_pose.position.z = 0.28

            # Orientation: quat(xyzw) = 0, 0, 0.229, 0.973
            # (Assuming missing x/y were 0 based on unit quaternion calculation)
            goal.initial_approach_pose.quaternion.x = 0.0
            goal.initial_approach_pose.quaternion.y = 0.229
            goal.initial_approach_pose.quaternion.z = 0
            goal.initial_approach_pose.quaternion.w = 0.973

            goal.retract_pose.CopyFrom(goal.initial_approach_pose)
            # Optional: Set defaults for other fields if needed
            # goal.approach_distance = 0.1
            # goal.lift_distance = 0.1

            if not self.publisher_action:
                 self.start_stream()

            self.publisher_action.put(goal.SerializeToString())
            logger.info(f"GraspAction Sent! ID: {action_id}")
            return True, action_id
        except Exception as e:
            import traceback
            logger.error(f"Failed to create/send action: {e} {traceback.format_exc()}")
            return False, str(e)

    # --- Getters ---
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

    def _ensure_model_loaded(self):
        if self.model is not None: return
        logger.info(f"Lazy Loading SAM3 on {self.device}...")
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
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise e

    def get_inference_state(self, image: Image.Image):
        self._ensure_model_loaded()
        if self.device == 'cuda': torch.cuda.empty_cache()
        return self.processor.set_image(image)

    @torch.inference_mode()
    def predict(self, inference_state, points, labels):
        self._ensure_model_loaded()
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

    def cleanup(self):
        self.model = None
        self.processor = None
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()

class SessionManager:
    def __init__(self):
        self.current_image = None
        self.inference_state = None
        self.current_mask = None
        self.current_overlay = None
    
    def set_image(self, image: Image.Image, inference_state):
        self.release_inference()
        self.current_image = image
        self.inference_state = inference_state
        self.current_mask = None
        self.current_overlay = None

    def release_inference(self):
        if self.inference_state and hasattr(self.inference_state, 'clear'):
            try: self.inference_state.clear()
            except: pass
        self.inference_state = None

# --- Flask App ---
app = Flask(__name__)
app.config.update(CONFIG)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_handler = SAM3ModelHandler()
session_manager = SessionManager()
zenoh_streamer = ZenohStreamer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_video', methods=['POST'])
def start_video():
    success = zenoh_streamer.start_stream()
    return jsonify({'success': success})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    zenoh_streamer.stop_stream()
    return jsonify({'success': True})

@app.route('/get_frame', methods=['GET'])
def get_frame():
    if not zenoh_streamer.is_active():
        return jsonify({'success': False, 'status': 'stopped'})
    img = zenoh_streamer.get_latest_frame()
    if not img:
        return jsonify({'success': True, 'status': 'waiting'})
    return jsonify({'success': True, 'status': 'streaming', 
                    'frame_data': ImageProcessor.to_base64(img)})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files: return jsonify({'error': 'No file'}), 400
    try:
        image = Image.open(request.files['image'].stream).convert("RGB")
        image = ImageProcessor.resize_if_needed(image, CONFIG['MAX_IMAGE_SIZE'])
        inf_state = model_handler.get_inference_state(image)
        session_manager.set_image(image, inf_state)
        return jsonify({'success': True, 'image_data': ImageProcessor.to_base64(image)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_mask', methods=['POST'])
def predict_mask():
    """Generates mask, PUBLISHES it to generate PC, and cleans up."""
    data = request.get_json()
    points = data.get('points', [])
    labels = data.get('labels', [])

    if not points: return jsonify({'error': 'No points'}), 400

    if session_manager.inference_state is None:
        if session_manager.current_image:
            inf_state = model_handler.get_inference_state(session_manager.current_image)
            session_manager.set_image(session_manager.current_image, inf_state)
        else:
            return jsonify({'error': 'No image context'}), 400

    try:
        points_arr = np.array(points)
        labels_arr = np.array(labels)
        masks, scores = model_handler.predict(session_manager.inference_state, points_arr, labels_arr)
        
        best_mask = masks[0]
        session_manager.current_mask = (best_mask > 0).astype(np.uint8) * 255
        
        # 1. Publish Mask -> Triggers Env to calculate Point Cloud
        zenoh_streamer.publish_mask(session_manager.current_mask)
        
        overlay_img = ImageProcessor.create_overlay(
            session_manager.current_image, best_mask, points_arr, labels_arr
        )
        session_manager.current_overlay = overlay_img

        # 2. Cleanup
        model_handler.cleanup()
        session_manager.release_inference()

        return jsonify({
            'success': True,
            'overlay_data': ImageProcessor.to_base64(overlay_img, "PNG")
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute_grasp', methods=['POST'])
def execute_grasp():
    """
    Formerly /download.
    Waits for mask_pc (triggered by predict_mask) and sends Grasp Action.
    """
    if session_manager.current_mask is None:
        return jsonify({'error': 'No mask generated yet.'}), 400

    # Wait for the Point Cloud to arrive from the environment
    success, result = zenoh_streamer.wait_and_publish_action(timeout=5.0)
    
    if success:
        return jsonify({'success': True, 'action_id': result, 'message': 'Grasp command sent!'})
    else:
        return jsonify({'error': result}), 504 # Gateway Timeout

@app.route('/select_place', methods=['POST'])
def select_place():
    # Placeholder for place logic if needed
    data = request.get_json()
    return jsonify({'success': True})

def signal_handler(sig, frame):
    model_handler.cleanup()
    zenoh_streamer.close()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    app.run(host='0.0.0.0', port=50052, debug=False)