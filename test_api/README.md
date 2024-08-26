# triton、flask、fastapi 接口测试

分别起一个推理模型的服务，模型推理平均时间为 0.0012

|             | 并发20   | 并发50   | 并发200  |
| ----------- | ------ | ------ | ------ |
| flask       | 0.0052 | 0.0046 | 0.0079 |
| fastapi     | 0.0024 | 0.0023 | 0.0023 |
| triton-http | 0.0020 | 0.0017 | 0.0015 |
| triton-grpc | 0.0028 | 0.0020 | 0.0018 |

测试发现：

1、triton对并发处理的很好，并发数越高，吞吐量越大，平均每条请求时间越少

2、fastapi的性能也很不错，比较稳定

3、flask性能相比前两个会比较弱，稳定性也不够

#### 起flask服务

```shell
python flask_server.py
```

#### 起fastapi服务

```shell
python fastapi_server.py
```

#### triton相关

起一个triton服务

```shell
tritonserver --model-repository=./triton_model
```

推理

```shell
python triton_http_request.py
python triton_grpc_request.py
```


