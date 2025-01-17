#pragma once
#include "InferenceInterface.h"
#include <onnxruntime_cxx_api.h>  // for ONNX Runtime C++ API

class ORTInfer : public InferenceInterface
{
private:
    Ort::Env env_;
    Ort::Session session_{ nullptr };
    std::vector<std::string> input_names_;  // Store input layer names
    std::vector<std::string> output_names_; // Store output layer names
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

public:
    std::string print_shape(const std::vector<std::int64_t>& v);
    ORTInfer(const std::string& model_path);
    size_t getSizeByDim(const std::vector<int64_t>& dims);

    std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;
};