#pragma once
#include "common.h"
#include "InferenceInterface.h"
#include "ORTInfer.h"

std::unique_ptr<InferenceInterface> setup_inference_engine(const std::string& weights, const std::string& modelConfiguration);