## yolov8 rust demo

1、下载 yolov8 onnx model  https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx

2、编译

```shell
cargo build --release
```

3、运行

```shell
./target/release/example-yolov8 --model_path --image_path
```


