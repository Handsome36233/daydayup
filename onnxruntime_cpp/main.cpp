#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "InferenceBackendSetup.h"
#include <iostream>
#include <vector>


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path> <image_path>\n";
        return -1;
    }

    std::string onnxModelPath = argv[1];
    std::string imgPath = argv[2];
    std::unique_ptr<InferenceInterface> engine;
    engine = setup_inference_engine(onnxModelPath, "");

    cv::Mat img = cv::imread(imgPath); // 读取图像
    cv::resize(img, img, cv::Size(128, 128));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    // 图像转为 tensor 并标准化
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(imgFloat, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] -= mean[c];    // 减去均值
        channels[c] /= std[c];     // 除以标准差
    }
    cv::merge(channels, imgFloat);

    cv::Mat blob;
    cv::dnn::blobFromImage(imgFloat, blob, 1.0, cv::Size(), cv::Scalar(), false, false);

    const auto[outputs, shapes] = engine->get_infer_results(blob);
    const std::any* output0 = outputs.front().data();
    const  std::vector<int64_t> shape0 = shapes.front();
    for (int i = 0; i < 2; i++) {
        std::cout << std::any_cast<float>(output0[i]) << std::endl;
    }
    
    return 0;
}
