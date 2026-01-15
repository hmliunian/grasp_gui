import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# -------------------------------
# 设备设置
# -------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------------
# 辅助显示函数
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

def show_masks(image, masks, points=None, labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca())
    if points is not None and labels is not None:
        show_points(points, labels, plt.gca())
    plt.axis('off')
    plt.show()

# -------------------------------
# 加载 SAM3 模型（仅 image 模型）
# -------------------------------
# 不使用 BPE 文件
CHECKPOINT = "/home/xuran-yao/code/DISCOVERSE_v2_exp/sam3/checkpoints/sam3.pt"
model = build_sam3_image_model(bpe_path=None,checkpoint_path=CHECKPOINT, enable_inst_interactivity=True)
model.to(device)
model.eval()

processor = Sam3Processor(model)

# -------------------------------
# 读取图片
# -------------------------------
image_path = "input_data/rgb_0000.png"  # <-- 替换为你的图片路径
image = Image.open(image_path).convert("RGB")

# -------------------------------
# 生成 image embedding
# -------------------------------
inference_state = processor.set_image(image)

# -------------------------------
# 指定点 prompt
# -------------------------------
# 例子：选择物体 + 背景
points = np.array([[520, 375], [1125, 625]])  # (x,y)
labels = np.array([1, 0])                     # 1=foreground, 0=background

# -------------------------------
# 预测 mask
# -------------------------------
masks, scores, logits = model.predict_inst(
    inference_state,
    point_coords=points,
    point_labels=labels,
    multimask_output=False
)

# -------------------------------
# 可视化结果
# -------------------------------
show_masks(image, masks, points, labels)

# -------------------------------
# 可选：保存 mask
# -------------------------------
mask_array = masks[0].astype(np.uint8) * 255
mask_image = Image.fromarray(mask_array)
mask_image.save("mask_output.png")
print("Mask saved as mask_output.png")
# 权重路径（请改成你下载的 sam3 权重）

