## yolov8 rust demo

项目来自 [ort/examples/yolov8 at main · pykeio/ort · GitHub](https://github.com/pykeio/ort/tree/main/examples/yolov8)

刚开始学习rust，我只是搬运工....

1、下载 yolov8 onnx model  https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx

2、编译

```shell
cargo build --release
```

3、运行

```shell
./target/release/example-yolov8 --model_path --image_path
```
