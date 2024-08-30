#include "InferenceBackendSetup.h"


std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration)
{
    return std::make_unique<ORTInfer>(weights);
}