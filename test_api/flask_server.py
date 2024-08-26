from flask import Flask, request, jsonify
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

app = Flask(__name__)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = image / 255.0  # 归一化
    return image

def run():
    with torch.no_grad():
        # 模型推理的逻辑在此处执行
        model(input_data)

@app.route("/test", methods=["POST"])
def test_apply():
    data = request.get_json()  # 获取请求数据
    result = {}
    try:
        # 这里可以使用数据进行模型推理
        run()
        result["message"] = 'success'
    except Exception as e:
        result["message"] = str(e)
    return jsonify(result)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "./assent/mug.jpg"
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # 添加 batch 维度
    input_data = torch.from_numpy(input_data).to(device)  # 转为 tensor
    model = models.resnet18(pretrained=True)
    model.to(device)
    model.eval()  # 设置模型为推理模式  
    app.run(host='0.0.0.0', port=6003)
