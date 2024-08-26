import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from multiprocessing import Process, Queue
import time

# 连接到 Triton Server
url = "localhost:8000"  # 服务器地址和端口
model_name = "resnet18_onnx"  # 模型名称

# 预处理图片
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = image / 255.0  # 归一化
    return image

# 推理函数
def infer():
    # 设置输入
    client = httpclient.InferenceServerClient(url=url)
    input_tensor = httpclient.InferInput('inputhaha', input_data.shape, 'FP32')
    input_tensor.set_data_from_numpy(input_data)
    # 设置输出
    output_tensor = httpclient.InferRequestedOutput('outputhaha')
    # 执行推理
    client.infer(model_name, inputs=[input_tensor], outputs=[output_tensor])


if __name__ == "__main__":
    image_path = "/workspace/images/mug.jpg"  # 输入图像文件路径
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # 添加 batch 维度
    num_processes = 200  # 并发进程数
    processes = []
    iters = 50
    t1 = time.time()
    for _ in range(iters):
        for _ in range(num_processes):
            p = Process(target=infer)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    t2 = time.time()
    print(f"time: {(t2 - t1)/num_processes/iters:.4f} seconds")

