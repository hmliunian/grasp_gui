# 外部接口集成文档

本文档说明如何与 SAM GUI 的外部接口进行集成。

## 工作流程

```
GUI → 外部服务: 发送图像（POST）
外部服务 → GUI: 发送处理后的 mask（POST）
```

## 接口1：接收图像（外部服务需要实现的接口）

GUI 会向外部服务发送图像，外部服务需要提供一个接口来接收。

### 请求格式

**URL**: 由外部服务提供（需要告知 GUI 开发人员）

**方法**: `POST`

**Content-Type**: `application/json`

**请求体**:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "width": 640,
  "height": 480
}
```

**字段说明**:
- `image`: Base64 编码的 PNG 图像，格式为 `data:image/png;base64,<base64_string>`
- `width`: 图像宽度（像素）
- `height`: 图像高度（像素）

### 响应格式

**成功响应** (HTTP 200):
```json
{
  "success": true,
  "message": "Image received successfully"
}
```

**错误响应** (HTTP 400/500):
```json
{
  "error": "Error description"
}
```

### 实现示例

```python
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/api/process_image', methods=['POST'])
def receive_image():
    data = request.get_json()
    image_data = data.get('image')
    
    # 解码 Base64 图像
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 在这里进行图像处理（生成 mask）
    # mask = your_processing_function(img)
    
    # 返回成功响应
    return jsonify({
        'success': True,
        'message': 'Image received successfully'
    }), 200
```

## 接口2：发送 Mask（外部服务调用 GUI 的接口）

外部服务处理完图像后，需要将生成的 mask 发送回 GUI。

### 请求格式

**URL**: `http://localhost:50052/external/receive_mask`（GUI 服务器地址）

**方法**: `POST`

**Content-Type**: `application/json`

**请求体**:
```json
{
  "mask_data": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**字段说明**:
- `mask_data`: Base64 编码的 PNG mask 图像，格式为 `data:image/png;base64,<base64_string>`

### Mask 格式要求

1. **格式**: PNG 图像
2. **颜色模式**: 灰度（单通道）
3. **像素值**: 
   - `0`: 背景区域
   - `255`: 前景/目标区域
   - 中间值（可选）: 可以表示置信度
4. **尺寸**: 可以是任意尺寸，GUI 会自动调整到与原始图像相同大小

### 响应格式

**成功响应** (HTTP 200):
```json
{
  "success": true,
  "message": "Mask received and stored in buffer"
}
```

**错误响应** (HTTP 400/500):
```json
{
  "error": "Error description"
}
```

### 实现示例

```python
import requests
import base64
from PIL import Image
import io

def send_mask_to_gui(mask_image, gui_url="http://localhost:50052"):
    """
    将 mask 发送到 GUI
    
    Args:
        mask_image: PIL Image 对象（灰度模式，L mode）
        gui_url: GUI 服务器地址
    """
    # 转换为 Base64
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mask_data = f"data:image/png;base64,{mask_base64}"
    
    # 发送请求
    response = requests.post(
        f"{gui_url}/external/receive_mask",
        headers={'Content-Type': 'application/json'},
        json={'mask_data': mask_data},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("Mask sent successfully")
        else:
            print(f"Failed: {result.get('message')}")
    else:
        print(f"HTTP Error: {response.status_code}")

# 使用示例
mask = Image.open('mask.png').convert('L')  # 确保是灰度图像
send_mask_to_gui(mask)
```

## 注意事项

1. **响应时间**: 建议外部服务在 10 秒内响应（GUI 的 timeout 为 10 秒）
2. **Mask 锁定**: 一旦 mask 在 GUI 中被用户确认，会进入锁定状态，不能再被替换
3. **异步处理**: 外部服务可以在后台异步处理图像，处理完成后随时可以发送 mask

## 测试示例

项目提供了两个测试脚本，用于验证接口功能：

### test_external_service_receiver.py

**功能**: 模拟外部服务，接收 GUI 发送的图像并保存到磁盘

**作用**:
- 启动一个 Flask 服务器（默认端口 50053）
- 监听 `/api/process_image` 接口
- 接收 GUI 发送的图像
- 将图像保存到 `test_outputs/received_images/` 目录
- 用于验证 GUI 是否正确发送图像

**使用方法**:
```bash
python test_external_service_receiver.py
```

### test_mask_sender.py

**功能**: 读取本地 mask 文件并发送到 GUI

**作用**:
- 读取指定的 mask 文件（PNG 格式）
- 将 mask 转换为 Base64 编码
- 发送 POST 请求到 GUI 的 `/external/receive_mask` 接口
- 用于验证外部服务能否正确发送 mask 到 GUI

**使用方法**:
```bash
# 发送单个 mask 文件
python test_mask_sender.py --mask_path path/to/mask.png

# 发送目录下所有 mask 文件
python test_mask_sender.py --mask_dir test_outputs/masks_to_send

# 指定 GUI 地址
python test_mask_sender.py --mask_path mask.png --gui_url http://localhost:50052
```
