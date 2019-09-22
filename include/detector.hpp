// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include <samples/ocv_common.hpp>

#include <inference_engine.hpp>

#include <opencv2/core/core.hpp>

#include "core.hpp"

struct BaseDetection
{
    InferenceEngine::ExecutableNetwork net;
    InferenceEngine::InferencePlugin *plugin;
    InferenceEngine::InferRequest::Ptr request;
    std::string topoName;
    std::string pathToModel;
    std::string deviceForInference;
    const size_t maxBatch;
    bool isBatchDynamic;
    const bool isAsync;
    mutable bool enablingChecked;
    mutable bool _enabled;
    const bool doRawOutputMessages;

    BaseDetection(std::string topoName,
                  const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync,
                  bool doRawOutputMessages);

    virtual ~BaseDetection();

    InferenceEngine::ExecutableNetwork *operator->();
    virtual InferenceEngine::CNNNetwork read() = 0;
    virtual void submitRequest();
    virtual void wait();
    bool enabled() const;
    void printPerformanceCounts();
};

struct FaceDetection : BaseDetection
{
    struct Result
    {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::string input;
    std::string output;
    double detectionThreshold;
    int maxProposalCount;
    int objectSize;
    int enquedFrames;
    float width;
    float height;
    float bb_enlarge_coefficient;
    float bb_dx_coefficient;
    float bb_dy_coefficient;
    bool resultsFetched;
    std::vector<std::string> labels;
    // std::vector<Result> results;
    TrackedObjects results;
    int frame_idx_ = -1;

    FaceDetection(const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync,
                  double detectionThreshold, bool doRawOutputMessages,
                  float bb_enlarge_coefficient, float bb_dx_coefficient,
                  float bb_dy_coefficient);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &frame, const int frame_idx);
    void fetchResults();
};

struct HeadPoseDetection : BaseDetection
{
    struct Results
    {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    std::string input;
    std::string outputAngleR;
    std::string outputAngleP;
    std::string outputAngleY;
    size_t enquedFaces;
    cv::Mat cameraMatrix;

    HeadPoseDetection(const std::string &pathToModel,
                      const std::string &deviceForInference,
                      int maxBatch, bool isBatchDynamic, bool isAsync,
                      bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    Results operator[](int idx) const;
};

struct AgeGenderDetection : BaseDetection
{
    struct Result
    {
        float age;
        float maleProb;
    };

    std::string input;
    std::string outputAge;
    std::string outputGender;
    size_t enquedFaces;

    AgeGenderDetection(const std::string &pathToModel,
                       const std::string &deviceForInference,
                       int maxBatch, bool isBatchDynamic, bool isAsync,
                       bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    Result operator[](int idx) const;
};

struct EmotionsDetection : BaseDetection
{
    std::string input;
    std::string outputEmotions;
    size_t enquedFaces;

    EmotionsDetection(const std::string &pathToModel,
                      const std::string &deviceForInference,
                      int maxBatch, bool isBatchDynamic, bool isAsync,
                      bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    std::map<int, float> operator[](int idx) const;

    static const std::vector<int> emotionsVec;
    static const std::vector<std::string> emotionsVecTxt;
};

struct FacialLandmarksDetection : BaseDetection
{
    std::string input;
    std::string outputFacialLandmarksBlobName;
    size_t enquedFaces;
    std::vector<std::vector<float>> landmarks_results;
    std::vector<cv::Rect> faces_bounding_boxes;

    FacialLandmarksDetection(const std::string &pathToModel,
                             const std::string &deviceForInference,
                             int maxBatch, bool isBatchDynamic, bool isAsync,
                             bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read() override;
    void submitRequest() override;

    void enqueue(const cv::Mat &face);
    std::vector<float> operator[](int idx) const;
};

struct Load
{
    BaseDetection &detector;

    explicit Load(BaseDetection &detector);

    void into(InferenceEngine::InferencePlugin &plg, bool enable_dynamic_batch = false) const;
};
