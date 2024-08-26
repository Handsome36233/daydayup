from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import time
from typing import Optional

app = FastAPI()

class Request(BaseModel):
    text: Optional[str] = None


# 设置允许的源（origin），可以是特定的域名或 "*" 代表允许所有
origins = [
    "*",                      # 允许所有来源
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 允许的源
    allow_credentials=True,           # 允许发送 Cookie
    allow_methods=["*"],              # 允许的 HTTP 方法
    allow_headers=["*"],              # 允许的 HTTP 头部
)


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = image / 255.0  # 归一化
    return image


def run():
    with torch.no_grad():
        model(input_data)

@app.post("/test")
async def test_apply(request: Request):
    data = request.dict()    
    result = {}
    try:
        run()
        result["message"] = 'success'
    except Exception as e:
        result["message"] = str(e)
    return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "./assent/mug.jpg"
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # 添加 batch 维度
    input_data = torch.from_numpy(input_data).to(device)  # 转为 tensor
    model = models.resnet18(pretrained=True)
    model.to(device)
    model.eval()  # 设置模型为推理模式  
    uvicorn.run(app, host='0.0.0.0', port=6003)
