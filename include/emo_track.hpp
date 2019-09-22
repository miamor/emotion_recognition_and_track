// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char video_message[] = "Required. Path to a video file or a folder with images "
                                    "(all images should have names 0000000001.jpg, 0000000002.jpg, etc).";

/// @brief message for model arguments
static const char face_detection_model_message[] = "Required. Path to the Pedestrian Detection Retail model (.xml) file.";
static const char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
static const char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
static const char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";
static const char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";

/// @brief message for assigning Pedestrian detection inference to device
static const char target_device_message_detection[] = "Optional. Specify the target device for face detection "
                                                      "(CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). ";

/// @brief Message for assigning head pose calculation to device
static const char target_device_message_hp[] = "Optional. Target device for Head Pose Estimation network "
                                               "(CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). ";

/// @brief Message for assigning age/gender calculation to device
static const char target_device_message_ag[] = "Optional. Target device for Age/Gender Recognition network (CPU, GPU, FPGA, HDDL, or MYRIAD). " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning emotions calculation to device
static const char target_device_message_em[] = "Optional. Target device for Emotions Recognition network (CPU, GPU, FPGA, HDDL, or MYRIAD). " \
"The demo will look for a suitable plugin for a specified device.";

/// @brief Message for assigning Facial Landmarks Estimation network to device
static const char target_device_message_lm[] = "Optional. Target device for Facial Landmarks Estimation network (CPU, GPU, FPGA, HDDL, or MYRIAD). " \
"The demo will look for a suitable plugin for device specified.";


/// @brief Message for the maximum number of simultaneously processed faces for Age Gender network
static const char num_batch_ag_message[] = "Optional. Number of maximum simultaneously processed faces for Age/Gender Recognition network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Head Pose network
static const char num_batch_hp_message[] = "Optional. Number of maximum simultaneously processed faces for Head Pose Estimation network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Emotions network
static const char num_batch_em_message[] = "Optional. Number of maximum simultaneously processed faces for Emotions Recognition network " \
"(by default, it is 16)";

/// @brief Message for the maximum number of simultaneously processed faces for Facial Landmarks Estimation network
static const char num_batch_lm_message[] = "Optional. Number of maximum simultaneously processed faces for Facial Landmarks Estimation network " \
"(by default, it is 16)";



/// @brief Message for dynamic batching support for HeadPose net
static const char dyn_batch_hp_message[] = "Optional. Enable dynamic batch size for Head Pose Estimation network";

/// @brief Message for dynamic batching support for AgeGender net
static const char dyn_batch_ag_message[] = "Optional. Enable dynamic batch size for Age/Gender Recognition network";

/// @brief Message for dynamic batching support for Emotions net
static const char dyn_batch_em_message[] = "Optional. Enable dynamic batch size for Emotions Recognition network";

/// @brief Message for dynamic batching support for Facial Landmarks Estimation network
static const char dyn_batch_lm_message[] = "Optional. Enable dynamic batch size for Facial Landmarks Estimation network";



/// @brief Message for probability threshold argument for face detections
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";


/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enable per-layer performance statistics.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "
                                           "Absolute path to the .xml file with the kernels description.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. "
                                                 "Absolute path to a shared library with the kernels implementation.";

/// @brief message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message for asynchronous mode
static const char async_message[] = "Optional. Enable asynchronous mode";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief message for delay between frames
static const char delay_message[] = "Optional. Delay between frames used for visualization. "
                                    "If negative, the visualization is turned off (like with the option 'no_show'). "
                                    "If zero, the visualization is made frame-by-frame.";

/// @brief message for output log
static const char output_log_message[] = "Optional. The file name to write output log file with results of face tracking. "
                                         "The format of the log file is compatible with MOTChallenge format.";

/// @brief message for the first frame
static const char first_frame_message[] = "Optional. The index of the first frame of video sequence to process. "
                                          "This has effect only if it is positive and the source video sequence is an image folder.";
/// @brief message for the last frame
static const char last_frame_message[] = "Optional. The index of the last frame of video sequence to process. "
                                         "This has effect only if it is positive and the source video sequence is an image folder.";

/// @brief Message for output video path
static const char output_video_message[] = "Optional. File to write output video with visualization to.";

/// @brief Message for debug argument
static const char debug_message[] = "Optional. Enable debug";


/// @brief Message for smooth argument
static const char no_smooth_output_message[] = "Optional. Do not smooth person attributes";

/// @brief Message for smooth argument
static const char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";

/// @brief Message for face enlarge coefficient argument
static const char bb_enlarge_coef_output_message[] = "Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face";

/// @brief Message for shifting coefficient by dx for detected faces
static const char dx_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Ox axis";

/// @brief Message for shifting coefficient by dy for detected faces
static const char dy_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Oy axis";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", video_message);


/// @brief Define parameter for face detection model file <br>
/// It is a required parameter
DEFINE_string(m_det, "", face_detection_model_message);

/// @brief Define parameter for facial landmark model file <br>
/// It is a required parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// \brief Define parameter for Face Detection  model file<br>
/// It is a optional parameter
DEFINE_string(m_ag, "", age_gender_model_message);

/// \brief Define parameter for Face Detection model file<br>
/// It is a optional parameter
DEFINE_string(m_em, "", emotions_model_message);

/// \brief Define parameter for Face Detection  model file<br>
/// It is a optional parameter
DEFINE_string(m_hp, "", head_pose_model_message);


/// @brief device the target device for face detection infer on <br>
DEFINE_string(d_det, "CPU", target_device_message_detection);

/// @brief device the target device for facial landnmarks regression infer on <br>
DEFINE_string(d_lm, "CPU", target_device_message_lm);

/// \brief Define parameter for target device for Age/Gender Recognition network<br>
DEFINE_string(d_ag, "CPU", target_device_message_ag);

/// \brief Define parameter for target device for Emotions Recognition network<br>
DEFINE_string(d_em, "CPU", target_device_message_em);

/// @brief Define parameter for target device for Head Pose Estimation network<br>
DEFINE_string(d_hp, "CPU", target_device_message_hp);


/// @brief Define probability threshold for face detections <br>
/// It is an optional parameter
DEFINE_double(t_det, 0.6, face_threshold_output_message);



/// \brief Define parameter for maximum batch size for Age/Gender Recognition network<br>
DEFINE_uint32(n_ag, 16, num_batch_ag_message);

/// \brief Define parameter to enable dynamic batch size for Age/Gender Recognition network<br>
DEFINE_bool(dyn_ag, false, dyn_batch_ag_message);

/// \brief Define parameter for maximum batch size for Head Pose Estimation network<br>
DEFINE_uint32(n_hp, 16, num_batch_hp_message);

/// \brief Define parameter to enable dynamic batch size for Head Pose Estimation network<br>
DEFINE_bool(dyn_hp, false, dyn_batch_hp_message);

/// \brief Define parameter for maximum batch size for Emotions Recognition network<br>
DEFINE_uint32(n_em, 16, num_batch_em_message);

/// \brief Define parameter to enable dynamic batch size for Emotions Recognition network<br>
DEFINE_bool(dyn_em, false, dyn_batch_em_message);

/// \brief Define parameter for maximum batch size for Facial Landmarks Estimation network<br>
DEFINE_uint32(n_lm, 16, num_batch_em_message);

/// \brief Define parameter to enable dynamic batch size for Facial Landmarks Estimation network<br>
DEFINE_bool(dyn_lm, false, dyn_batch_em_message);



/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// \brief Define a flag to enable aynchronous execution<br>
/// It is an optional parameter
DEFINE_bool(async, false, async_message);

/// @brief Define delay for visualization <br>
/// It is an optional parameter
DEFINE_int32(delay, 3, delay_message);

/// @brief Define output log path to store tracking results <br>
/// It is an optional parameter
DEFINE_string(out, "", output_log_message);

/// @brief Define the first frame to process <br>
/// It is an optional parameter
DEFINE_int32(first, -1, first_frame_message);

/// @brief Define the last frame to process <br>
/// It is an optional parameter
DEFINE_int32(last, -1, last_frame_message);

/// @brief Flag to output raw pipeline results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

DEFINE_bool(debug, false, debug_message);
DEFINE_bool(draw_track, false, debug_message);


/// \brief Define a flag to disable smoothing person attributes<br>
/// It is an optional parameter
DEFINE_bool(no_smooth, false, no_smooth_output_message);

DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);


/// \brief Define a parameter to shift face bounding box by Ox for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(dx_coef, 1, dx_coef_output_message);

/// \brief Define a parameter to shift face bounding box by Oy for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(dy_coef, 1, dy_coef_output_message);

/// \brief Define a parameter to enlarge the bounding box around the detected face for more robust operation of face analytics networks<br>
/// It is an optional parameter
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_message);


/**
 * @brief This function show a help message
 */
static void showUsage()
{
    std::cout << std::endl;
    std::cout << "face_reg_track [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                            " << help_message << std::endl;
    std::cout << "    -i \"<path>\"                 " << video_message << std::endl;
    std::cout << "    -m_det \"<path>\"             " << face_detection_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"              " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_ag \"<path>\"              " << age_gender_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"              " << emotions_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"              " << head_pose_model_message << std::endl;
    std::cout << "    -l \"<absolute_path>\"        " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"        " << custom_cldnn_message << std::endl;
    std::cout << "    -d_det \"<device>\"           " << target_device_message_detection << std::endl;
    std::cout << "    -d_lm \"<device>\"            " << target_device_message_lm << std::endl;
    std::cout << "    -d_ag \"<device>\"            " << target_device_message_ag << std::endl;
    std::cout << "    -d_em \"<device>\"            " << target_device_message_em << std::endl;
    std::cout << "    -d_hp \"<device>\"            " << target_device_message_hp << std::endl;
    std::cout << "    -pc                           " << performance_counter_message << std::endl;
    std::cout << "    -no_show                      " << no_show_processed_video << std::endl;
    std::cout << "    -delay                        " << delay_message << std::endl;
    std::cout << "    -first                        " << first_frame_message << std::endl;
    std::cout << "    -last                         " << last_frame_message << std::endl;
    std::cout << "    -debug                        " << debug_message << std::endl;
    std::cout << "    -r                             " << raw_output_message << std::endl;
    std::cout << "    -out \"<path>\"               " << output_log_message << std::endl;
}
