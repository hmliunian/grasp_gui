import os
import base64
import io
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# ========================
# Config
# ========================
SAVE_DIR = "/home/wm/Documents/roombia/compute/third_party/grasp_gui/static/uploads/ming_receive"
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI(title="Mock Mask Receiver")


# ========================
# Data model
# ========================
class ExternalMaskData(BaseModel):
    mask_data: str


# ========================
# Utils
# ========================
def decode_base64_image(data_url: str) -> Image.Image:
    """
    Decode data:image/png;base64,... to PIL Image
    """
    if data_url.startswith("data:image"):
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_bytes))


# ========================
# API
# ========================
@app.post("/external/receive_mask")
async def receive_mask(data: ExternalMaskData):
    try:
        if not data.mask_data:
            return JSONResponse(
                status_code=400,
                content={"error": "mask_data not provided"}
            )

        # Decode
        mask_img = decode_base64_image(data.mask_data).convert("L")

        # Generate filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"mask_{ts}.png"
        save_path = os.path.join(SAVE_DIR, filename)

        # Save
        mask_img.save(save_path)

        return {
            "success": True,
            "message": "Mask received and saved",
            "path": save_path,
            "size": mask_img.size
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ========================
# Entry
# ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=50056,
        reload=False
    )
