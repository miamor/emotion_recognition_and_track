// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core.hpp"

#include <iostream>

const int TrackedObject::UNKNOWN_LABEL_IDX = -1;


bool operator==(const TrackedObject& first, const TrackedObject& second) {
    return ( (first.rect == second.rect)
            && (first.confidence == second.confidence)
            && (first.frame_idx == second.frame_idx)
            && (first.object_id == second.object_id)
            // && (first.label == second.label)
            && (first.timestamp == second.timestamp) );
}

bool operator!=(const TrackedObject& first, const TrackedObject& second) {
    return !(first == second);
}
