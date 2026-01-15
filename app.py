import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io

# 你的 SAM3 推理代码
from sam3_inference import run_sam3  # 你自己实现的 SAM3 推理逻辑

st.title("￼ SAM3 Web Segmentation")

# 图片上传
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始图像", use_column_width=True)

    # Canvas 供用户点击标注点
    st.write("￼ 点击画布标注 positive/negative 点（Shift+点击为 negative）")

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)", 
        stroke_width=3,
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="sam_canvas",
    )

    # 点击点记录
    points = []
    if canvas.json_data is not None:
        for obj in canvas.json_data["objects"]:
            x, y = obj["left"], obj["top"]
            label = 1  # 你可以根据 Shift 键等设定 negative/positive
            points.append([x, y, label])

    st.write("标注点：", points)

    # 按钮触发分割
    if st.button("生成 Mask"):
        mask = run_sam3(image, points)  # 返回 numpy array mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        st.image(mask_img, caption="输出 Mask", use_column_width=True)