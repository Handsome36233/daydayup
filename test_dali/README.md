# dali 测试

#### DALI项目地址

https://github.com/NVIDIA/DALI

用gpu去做数据前处理，测试下对应的性能

显卡：RTX 4070 Ti SUPER

#### 测试 读取图片

（A）opencv+torchvision的dataloader 

（B）dali

|       | batch=1 | batch=16<br/>num_threads=1 | batch=16<br/>num_threads=4 | batch=16<br/>num_threads=8 |
| ----- | ------- | -------------------------- | -------------------------- | -------------------------- |
| A-cpu | 0.0370  | 0.0355                     | 0.0908                     | 0.1201                     |
| B-cpu | 0.0111  | 0.0136                     | 0.0035                     | 0.0031                     |
| A-gpu | 0.0432  | 0.0448                     | 0.0864                     | 0.1680                     |
| B-gpu | 0.0092  | 0.0081                     | 0.0032                     | 0.0016                     |

1、使用dali可以加快数据前处理，效率提升非常大，使用gpu更加明显。

2、torchvision的dataloader随着num_threads增大，效率反而变慢了，这个可能是因为测试的数据量太小导致。

3、torchvision的dataloader将数据变成gpu的tensor需要额外消耗时间。

#### 测试 dali的图像前处理

resize 到 256x256

|             | batch=1 | batch=16  <br>num_threads=1 | batch=16  <br>num_threads=4 | batch=16  <br>num_threads=8 |
| ----------- | ------- | --------------------------- | --------------------------- | --------------------------- |
| base-cpu    | 0.0111  | 0.0136                      | 0.0035                      | 0.0031                      |
| base-gpu    | 0.0092  | 0.0081                      | 0.0032                      | 0.0016                      |
| +resize-cpu | 0.0443  | 0.0450                      | 0.0114                      | 0.0077                      |
| +resize-gpu | 0.0093  | 0.0005                      | 0.00049                     | 0.00048                     |

resize 到 2560x2560

|             | batch=1 | batch=16  <br>num_threads=1 | batch=16  <br>num_threads=4 | batch=16  <br>num_threads=8 |
| ----------- | ------- | --------------------------- | --------------------------- | --------------------------- |
| +resize-cpu | 0.2957  | 0.3326                      | 0.085                       | 0.0471                      |
| +resize-gpu | 0.0086  | 0.0029                      | 0.0029                      | 0.0029                      |

1、使用gpu做resize性能提升非常大。

2、还有很多前处理算子都能用gpu提速，具体可参考 [Image Processing &#8212; NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/image_processing/index.html)