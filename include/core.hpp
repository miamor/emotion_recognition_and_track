// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>

struct LogObject
{
    int object_id;
    int frame_start;
    int frame_end;
    double tp_start;
    double tp_end;
    std::vector<int> emos;

    LogObject(int object_id,
              int frame_start, int frame_end,
              double tp_start, double tp_end,
              std::vector<int> emos) : 
              object_id(object_id), 
              frame_start(frame_start), frame_end(frame_end), 
              tp_start(tp_start), tp_end(tp_end), 
              emos(emos) {}
};

///
/// \brief The TrackedObject struct defines properties of detected object.
///
struct TrackedObject
{
    cv::Rect rect;      ///< Detected object ROI (zero area if N/A).
    double confidence;  ///< Detection confidence level (-1 if N/A).
    int frame_idx;      ///< Frame index where object was detected (-1 if N/A).
    int object_id;      ///< Unique object identifier (-1 if N/A).
    uint64_t timestamp; ///< Timestamp in milliseconds.

    int emo;

    int label; // either id of a label, or UNKNOWN_LABEL_IDX
    std::string label_txt;
    double dist; // Kuhn distance

    static const int UNKNOWN_LABEL_IDX; // the value (-1) for unknown label

    TrackedObject(const cv::Rect &rect = cv::Rect(), float conf = -1.0f,
                  int label = -1, int object_id = -1,
                  std::string label_txt = "", double dist = 999,
                  int frame_idx = -1, int emo = -1)
        : rect(rect), confidence(conf),
          object_id(object_id), label(label),
          label_txt(label_txt), dist(dist),
          frame_idx(frame_idx), emo(emo) {}

    // ///
    // /// \brief Default constructor.
    // ///
    // TrackedObject()
    //     : confidence(-1),
    //     frame_idx(-1),
    //     object_id(-1),
    //     label(-1),
    //     timestamp(0) {}

    // ///
    // /// \brief Constructor with parameters.
    // /// \param rect Bounding box of detected object.
    // /// \param confidence Confidence of detection.
    // /// \param frame_idx Index of frame.
    // /// \param object_id Object ID.
    // ///
    // TrackedObject(const cv::Rect &rect, float confidence,
    //               int frame_idx,
    //               int label,
    //               int object_id)
    //     : rect(rect),
    //     confidence(confidence),
    //     frame_idx(frame_idx),
    //     object_id(object_id),
    //     label(label),
    //     timestamp(0) {}
};

// using TrackedObjects = std::deque<TrackedObject>;

using TrackedObjects = std::vector<TrackedObject>;

bool operator==(const TrackedObject &first, const TrackedObject &second);
bool operator!=(const TrackedObject &first, const TrackedObject &second);
/// (object id, detected objects) pairs collection.
using ObjectTracks = std::unordered_map<int, TrackedObjects>;
