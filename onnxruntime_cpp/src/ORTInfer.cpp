#include "ORTInfer.h"

ORTInfer::ORTInfer(const std::string& model_path) : InferenceInterface{model_path, ""}
{
    env_=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnx Runtime Inference");
    Ort::SessionOptions session_options;

    session_options = Ort::SessionOptions();
    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session_.GetInputCount(); i++)
    {
        input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
        auto input_shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto input_type = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        std::cout << "\t" << input_names_.at(i) << " : " << print_shape(input_shapes)<<std::endl;
        input_shapes[0] = input_shapes[0] == -1 ? 1 : input_shapes[0];
        input_shapes_.emplace_back(input_shapes);

        std::string input_type_str;
        switch (input_type)
        {
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                input_type_str = "Float";
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                input_type_str = "Int64";
                break;
            // Handle other data types as needed
            default:
                input_type_str = "Unknown";
        }
    }

    const auto network_width = static_cast<int>(input_shapes_[0][3]);
    const auto network_height = static_cast<int>(input_shapes_[0][2]);
    const auto channels = static_cast<int>(input_shapes_[0][1]);

    std::cout << "channels " << channels << std::endl;
    std::cout << "width " << network_width << std::endl;
    std::cout << "height " << network_height << std::endl;

    // Print name/shape of outputs
    std::cout << "Output Node Name/Shape (" << session_.GetOutputCount() << "):" << std::endl;
    for (std::size_t i = 0; i < session_.GetOutputCount(); i++)
    {
        output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names_.at(i) << " : " << print_shape(output_shapes)<<std::endl;
        output_shapes_.emplace_back(output_shapes);
    }
}

std::string ORTInfer::print_shape(const std::vector<std::int64_t>& v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

size_t ORTInfer::getSizeByDim(const std::vector<int64_t>& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i)
    {
        if (dims[i] == -1 || dims[i] == 0)
        {
            continue;
        }
        size *= dims[i];
    }
    return size;
}

std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> ORTInfer::get_infer_results(const cv::Mat& input_blob)
{
    std::vector<std::vector<std::any>> outputs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<float>> input_tensors(session_.GetInputCount());
    std::vector<Ort::Value> in_ort_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int64_t> orig_target_sizes;

    input_tensors[0] = blob2vec(input_blob);
    in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensors[0].data(),
        getSizeByDim(input_shapes_[0]),
        input_shapes_[0].data(),
        input_shapes_[0].size()
    ));

    // RTDETR case, two inputs
    if (input_tensors.size() > 1)
    {
        orig_target_sizes = { static_cast<int64_t>(input_blob.size[2]), static_cast<int64_t>(input_blob.size[3]) };
        // Assuming input_tensors[1] is of type int64
        in_ort_tensors.emplace_back(Ort::Value::CreateTensor<int64>(
            memory_info,
            orig_target_sizes.data(),
            getSizeByDim(orig_target_sizes),
            input_shapes_[1].data(),
            input_shapes_[1].size()
        ));
    }

    // Run inference
    std::vector<const char*> input_names_char(input_names_.size());
    std::transform(input_names_.begin(), input_names_.end(), input_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names_.size());
    std::transform(output_names_.begin(), output_names_.end(), output_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<Ort::Value> output_ort_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names_char.data(),
        in_ort_tensors.data(),
        in_ort_tensors.size(),
        output_names_char.data(),
        output_names_.size()
    );

    // Process output tensors
    assert(output_ort_tensors.size() == output_names_.size());

    for (const Ort::Value& output_tensor : output_ort_tensors)
    {
        const auto& shape_ref = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> shape(shape_ref.begin(), shape_ref.end());

        size_t num_elements = 1;
        for (int64_t dim : shape) {
            num_elements *= dim;
        }

        std::vector<std::any> tensor_data;

        // Retrieve tensor data
        const int onnx_type = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();
        switch (onnx_type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* output_data_float = output_tensor.GetTensorData<float>();
            tensor_data.reserve(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                tensor_data.emplace_back(output_data_float[i]);
            }
            break;
        }
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t* output_data_int64 = output_tensor.GetTensorData<int64_t>();
            tensor_data.reserve(num_elements);
            for (size_t i = 0; i < num_elements; ++i) {
                tensor_data.emplace_back(output_data_int64[i]);
            }
            break;
        }
        // Add cases for other data types as needed
        default:
            std::exit(1);
        }

        // Store the tensor data in outputs
        outputs.emplace_back(tensor_data);
        shapes.emplace_back(shape);
    }

    return std::make_tuple(outputs, shapes);
}