import onnxruntime as ort
import torch
import torchvision.transforms as transforms
import cv2


def inference(image_path: str, onnx_model_path: str):
    ort_session = ort.InferenceSession(onnx_model_path)
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为 tensor 并标准化
    image_tensor = transforms.functional.to_tensor(image)
    image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]
    image_tensor = image_tensor[None, ...].numpy()  # 转为 Numpy 格式
    # 推理
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(ort_outputs)


onnx_model_path = 'cls_0120.onnx'
img_path = "image.png"
# 模型推理
inference(img_path, onnx_model_path)
