// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono> // NOLINT

#include "core.hpp"
#include "utils.hpp"
#include "tracker.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "detector.hpp"
#include "image_reader.hpp"
#include "emo_track.hpp"
#include "face.hpp"
#include "visualizer.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gflags/gflags.h>
#include <samples/slog.hpp>

using namespace InferenceEngine;
using ImageWithFrameIndex = std::pair<cv::Mat, int>;

// C library headers
#include <stdio.h>
#include <string.h>

// Linux headers
#include <fcntl.h>   // Contains file controls like O_RDWR
#include <errno.h>   // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h>  // write(), read(), close()

int set_interface_attribs(int fd, int speed, int parity)
{
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0)
    {
        printf("error %d from tcgetattr", errno);
        return -1;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit chars
    // disable IGNBRK for mismatched speed tests; otherwise receive break
    // as \000 chars
    tty.c_iflag &= ~IGNBRK; // disable break processing
    tty.c_lflag = 0;        // no signaling chars, no echo,
                            // no canonical processing
    tty.c_oflag = 0;        // no remapping, no delays
    tty.c_cc[VMIN] = 0;     // read doesn't block
    tty.c_cc[VTIME] = 5;    // 0.5 seconds read timeout

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

    tty.c_cflag |= (CLOCAL | CREAD);   // ignore modem controls,
                                       // enable reading
    tty.c_cflag &= ~(PARENB | PARODD); // shut off parity
    tty.c_cflag |= parity;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
    {
        printf("error %d from tcsetattr", errno);
        return -1;
    }
    return 0;
}

void set_blocking(int fd, int should_block)
{
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0)
    {
        printf("error %d from tggetattr", errno);
        return;
    }

    tty.c_cc[VMIN] = should_block ? 1 : 0;
    tty.c_cc[VTIME] = 5; // 0.5 seconds read timeout

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
        printf("error %d setting term attributes", errno);
}


std::unique_ptr<Tracker>
CreatePedestrianTracker(bool should_keep_tracking_info)
{
    TrackerParams params;

    if (should_keep_tracking_info)
    {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<Tracker> tracker(new Tracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(
            cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast =
        std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    std::cout << "WARNING: Either reid model or reid weights "
              << "were not specified. "
              << "Only fast reidentification approach will be used." << std::endl;

    return tracker;
}

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty())
    {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_det.empty())
    {
        throw std::logic_error("Parameter -m_det is not set");
    }

    return true;
}



int main_work(int argc, char **argv)
{
    int serial_port = open("/dev/ttyUSB0", O_RDWR);

    if (serial_port < 0)
    {
        printf("Error %i from open: %s\n", errno, strerror(errno));
    }
    set_interface_attribs(serial_port, B115200, 0);
    set_blocking(serial_port, 1);
    //char *s = "ON";
    //write(serial_port, s, sizeof(s));


    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    if (!ParseAndCheckCommandLine(argc, argv))
    {
        return 0;
    }

    const char ESC_KEY = 27;
    const cv::Scalar blue_color(255, 0, 0);
    // const cv::Scalar green_color(0, 255, 0);
    const cv::Scalar green_color(11, 138, 100);
    const cv::Scalar red_color(0, 0, 255);
    const cv::Scalar white_color(255, 255, 255);

    // Reading command line parameters.
    auto video_path = FLAGS_i;

    const auto det_model = FLAGS_m_det;
    const auto det_weights = fileNameNoExt(FLAGS_m_det) + ".bin";

    const auto outpath_log_track = (!FLAGS_out.empty() ? FLAGS_out + "/log_track.txt" : "");
    const auto outpath_log = (!FLAGS_out.empty() ? FLAGS_out + "/log.txt" : "");

    auto custom_cpu_library = FLAGS_l;
    auto path_to_custom_layers = FLAGS_c;
    bool should_use_perf_counter = FLAGS_pc;

    bool should_show = !FLAGS_no_show;
    int delay = FLAGS_delay;
    if (!should_show)
        delay = -1;
    should_show = (delay >= 0);

    int first_frame = FLAGS_first;
    int last_frame = FLAGS_last;

    bool should_save_det_log = !outpath_log_track.empty();

    if (first_frame >= 0)
        std::cout << "first_frame = " << first_frame << std::endl;
    if (last_frame >= 0)
        std::cout << "last_frame = " << last_frame << std::endl;

    std::vector<std::string> devices{FLAGS_d_det, FLAGS_d_lm, FLAGS_d_em, FLAGS_d_hp};
    std::map<std::string, InferencePlugin> plugins_for_devices =
        LoadPluginForDevices(
            devices, custom_cpu_library, path_to_custom_layers,
            should_use_perf_counter);

    // detection::DetectorConfig detector_config(det_model, det_weights);
    // detector_config.confidence_threshold = FLAGS_t_det;
    // auto detector_plugin = plugins_for_devices[FLAGS_d_det];
    // detection::ObjectDetector face_detector(detector_config, detector_plugin);

    FaceDetection faceDetector(FLAGS_m_det, FLAGS_d_det, 1, false, FLAGS_async, FLAGS_t_det, FLAGS_r,
                               static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
    AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async, FLAGS_r);
    EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async, FLAGS_r);
    FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async, FLAGS_r);

    Load(faceDetector).into(plugins_for_devices[FLAGS_d_det], false);
    Load(ageGenderDetector).into(plugins_for_devices[FLAGS_d_ag], FLAGS_dyn_ag);
    Load(emotionsDetector).into(plugins_for_devices[FLAGS_d_em], FLAGS_dyn_em);
    Load(facialLandmarksDetector).into(plugins_for_devices[FLAGS_d_lm], FLAGS_dyn_lm);

    bool should_keep_tracking_info = should_save_det_log;
    std::unique_ptr<Tracker> tracker =
        CreatePedestrianTracker(should_keep_tracking_info);

    // Opening video.
    std::unique_ptr<ImageReader> video =
        ImageReader::CreateImageReaderForPath(video_path);

    PT_CHECK(video->IsOpened()) << "Failed to open video: " << video_path;
    double video_fps = video->GetFrameRate();

    if (first_frame > 0)
        video->SetFrameIndex(first_frame);

    if (should_show)
    {
        std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
    }

    std::ofstream outputFile(outpath_log);

    float work_time_ms = 0.f;
    size_t work_num_frames = 0;

    std::vector<LogObject> logObjs;

    std::ostringstream out;
    std::list<Face::Ptr> faces;
    size_t id = 0;

    cv::VideoWriter videoWriter;
    if (!FLAGS_out.empty())
    {
        videoWriter.open(FLAGS_out + "/out_emo_track.avi", cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(640, 480));
    }
    Visualizer::Ptr visualizer;
    if (!FLAGS_no_show || !FLAGS_out.empty())
    {
        visualizer = std::make_shared<Visualizer>(cv::Size(640, 480));
        if (!FLAGS_no_show_emotion_bar && emotionsDetector.enabled())
        {
            visualizer->enableEmotionBar(emotionsDetector.emotionsVecTxt, emotionsDetector.emotionsVec);
        }
    }

    std::cout<<"---FIRST STEP---"<<std::endl;
    std::cout<<"Write GREEN to serial port"<<std::endl;
    char *s = "G";
    write(serial_port, s, 1);

    for (;;)
    {
        // char *s = "OFF";
        // std::cout << s <<  "-------------------------------------------------------------" << std::endl;
        // write(serial_port, s, sizeof(s));
        auto started = std::chrono::high_resolution_clock::now();

        auto pair = video->Read();
        cv::Mat frame = pair.first;

        const cv::Mat frame_cp = frame.clone();
        const cv::Mat frame_copy = frame.clone();

        int frame_idx = pair.second;

        if (frame.empty())
            break;

        // cv::imwrite(FLAGS_out+"/frames/"+std::to_string(work_num_frames)+".png", frame);

        PT_CHECK(frame_idx >= first_frame);

        if ((last_frame >= 0) && (frame_idx > last_frame))
        {
            std::cout << "Frame " << frame_idx << " is greater than last_frame = "
                      << last_frame << " -- break";
            break;
        }

        // face_detector.submitFrame(frame, frame_idx);
        // face_detector.waitAndFetchResults();

        // TrackedObjects detections = face_detector.getResults();

        faceDetector.enqueue(frame, work_num_frames);
        faceDetector.submitRequest();
        faceDetector.wait();
        faceDetector.fetchResults();
        TrackedObjects detections = faceDetector.results;

        if (detections.size() > 0)
        {
            // for (const auto &detection : detections)
            for (size_t i = 0; i < detections.size(); i++)
            {
                const auto &detection = detections[i];

                cv::Mat face = frame_cp(detection.rect);
                ageGenderDetector.enqueue(face);
                emotionsDetector.enqueue(face);
                facialLandmarksDetector.enqueue(face);
            }

            ageGenderDetector.submitRequest();
            emotionsDetector.submitRequest();
            facialLandmarksDetector.submitRequest();

            ageGenderDetector.wait();
            emotionsDetector.wait();
            facialLandmarksDetector.wait();
        }

        faces.clear();
        for (size_t i = 0; i < detections.size(); i++)
        {
            const auto &detection = detections[i];

            cv::Rect rect = detection.rect;

            Face::Ptr face;
            face = std::make_shared<Face>(id++, rect);

            face->ageGenderEnable((ageGenderDetector.enabled() &&
                                   i < ageGenderDetector.maxBatch));
            if (face->isAgeGenderEnabled())
            {
                AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                face->updateGender(ageGenderResult.maleProb);
                face->updateAge(ageGenderResult.age);
            }

            face->emotionsEnable((emotionsDetector.enabled() &&
                                  i < emotionsDetector.maxBatch));
            if (face->isEmotionsEnabled())
            {
                face->updateEmotions(emotionsDetector[i]);
                detections[i].emo = face->getMainEmotion().first;
            }

            face->landmarksEnable((facialLandmarksDetector.enabled() &&
                                   i < facialLandmarksDetector.maxBatch));
            if (face->isLandmarksEnabled())
            {
                face->updateLandmarks(facialLandmarksDetector[i]);
            }

            faces.push_back(face);
        }

        // timestamp in milliseconds
        uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_idx);
        tracker->Process(frame, detections, cur_timestamp);

        // auto tracked_faces = tracker->TrackedDetections();
        auto tracked_faces = tracker->TrackedDetectionsWithLabels();

        //  Visualizing results
        if (!FLAGS_no_show)
        {
            // Drawing colored "worms" (tracks)
            if (FLAGS_draw_track)
            {
                frame = tracker->DrawActiveTracks(frame);
            }

            out.str("");
            //out << "Total image throughput: " << std::fixed << std::setprecision(2) << 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
            cv::putText(frame, out.str(), cv::Point2f(10, 45), cv::FONT_HERSHEY_TRIPLEX, 1.2, cv::Scalar(255, 0, 0), 2);

            // drawing faces
            visualizer->draw(frame, faces);

            if (!FLAGS_no_show)
            {
                cv::imshow("Detection results", frame);
            }
        }

        // Save log
        if (should_save_det_log)
        {
            DetectionLog log = tracker->GetDetectionLog(true);
            SaveDetectionLogToTrajFile(outpath_log_track, log, logObjs, video_fps, serial_port);
        }

        ++work_num_frames;

        char k = cv::waitKey(delay);
        if (k == ESC_KEY)
            break;


        auto elapsed = std::chrono::high_resolution_clock::now() - started;
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        work_time_ms += elapsed_ms;

        // std::cout<< "time: " << work_time_ms << std::endl;

    }

    // if (should_keep_tracking_info)
    // {
    //     DetectionLog log = tracker->GetDetectionLog(true);

    //     if (should_save_det_log)
    //         SaveDetectionLogToTrajFile(outpath_log_track, log, logObjs, video_fps, serial_port);
    // }

    close(serial_port);

    if (should_use_perf_counter)
    {
        // face_detector.PrintPerformanceCounts();
        tracker->PrintReidPerformanceCounts();
    }
    return 0;
}

int main(int argc, char **argv)
{
    try
    {
        main_work(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;

    return 0;
}
