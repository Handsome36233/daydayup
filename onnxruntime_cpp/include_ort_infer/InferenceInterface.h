#pragma once
#include "common.h"

class InferenceInterface{
    	
    public:
        InferenceInterface(const std::string& weights, const std::string& modelConfiguration)
        {

        }

        
        virtual std::tuple<std::vector<std::vector<std::any>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

    protected:
        std::vector<float> blob2vec(const cv::Mat& input_blob);

};