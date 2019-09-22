// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <ie_plugin_config.hpp>

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <ext_list.hpp>

#include <unistd.h>  // write(), read(), close()

using namespace InferenceEngine;

float round_(float var)
{
    // 37.66666 * 100 =3766.66
    // 3766.66 + .5 =37.6716    for rounding off value
    // then type cast to int so value is 3766
    // then divided by 100 so the value converted into 37.66
    float value = (int)(var * 100 + .5);
    return (float)value / 100;
}

namespace
{

template <typename StreamType>
void SaveDetectionLogToStream(StreamType &stream,
                              const DetectionLog &log)
{

    for (const auto &entry : log)
    {
        std::vector<TrackedObject> objects(entry.objects.begin(),
                                           entry.objects.end());
        std::sort(objects.begin(), objects.end(),
                  [](const TrackedObject &a,
                     const TrackedObject &b) { return a.object_id < b.object_id; });
        for (const auto &object : objects)
        {
            auto frame_idx_to_save = entry.frame_idx;
            stream << frame_idx_to_save << ",";
            stream << object.object_id << ","
                   << object.rect.x << "," << object.rect.y << ","
                   << object.rect.width << "," << object.rect.height << ","
                   << object.label_txt << "," << object.dist << "," << object.timestamp;
            stream << "\n";
        }
    }
}

template <typename StreamType>
void SaveDetectionLogToStream(StreamType &stream,
                              const DetectionLog &log,
                              std::vector<LogObject> &logObjs,
                              const double fps,
                              const int serial_port)
{

    std::vector<int> emos;

    for (const auto &entry : log)
    {

        std::vector<TrackedObject> objects(entry.objects.begin(),
                                           entry.objects.end());
        std::sort(objects.begin(), objects.end(),
                  [](const TrackedObject &a,
                     const TrackedObject &b) { return a.object_id < b.object_id; });

        for (size_t o = 0; o < objects.size(); o++)
        {

            const auto &object = objects[o];

            bool append = true;
            for (size_t i = 0; i < logObjs.size(); i++)
            {
                auto log_obj = logObjs[i];
                if (log_obj.object_id == object.object_id)
                {
                    append = false;
                    
                    logObjs[i].frame_end = object.frame_idx;
                    logObjs[i].tp_end = (double)object.frame_idx / fps;

                    emos.push_back(object.emo);
                    logObjs[i].emos = emos;
                }
            }
            if (append) {
                emos.push_back(object.emo);
                logObjs.emplace_back(object.object_id, object.frame_idx, object.frame_idx, (double)object.frame_idx / fps, (double)object.frame_idx / fps, emos);
            }
        }
    }


    char *s = "";
    for (const auto &logObj : logObjs)
    {
        stream << logObj.object_id << ","
               << logObj.frame_start << "-" << logObj.frame_end << ","
               << logObj.tp_start << "-" << logObj.tp_end 
               << "("<<logObj.emos.size()<<")[";
        
        /* Debug emo list */
        int fr_ = 0;
        int segment = 0;
        for (const auto & emo : logObj.emos) {
            if (fr_ > 0 || segment > 0) stream << "-";
            stream << emo;
            fr_++;
        }
        
        stream << "]\n";



        int fr = 0;
        int continuous_angry_or_sad = 0;
        bool is_continuous_angry_or_sad = true;
        int continuous_happy = 0;
        bool is_continuous_happy = true;

        if (logObj.emos.size() > 10) {
            std::cout<<"logObj.emos.size()="<<logObj.emos.size()<<std::endl;
            
            // loop backwards. we consider only last 10 frames
            for (auto it = logObj.emos.rbegin(); it != logObj.emos.rend(); ++it ) {
                auto emo = *it;
                fr++;

                std::cout<<"\t"<<emo<<std::endl;
                
                // count angry
                if (emo == 4 || emo == 2) {
                    continuous_angry_or_sad++;
                    is_continuous_angry_or_sad = true;
                } else {
                    continuous_angry_or_sad = 0;
                    is_continuous_angry_or_sad = false;
                }
                // count happy
                if (emo == 1) {
                    continuous_happy++;
                    is_continuous_happy = true;
                } else {
                    continuous_happy = 0;
                    is_continuous_happy = false;
                }

                if (continuous_angry_or_sad == 10) {
                    if (s != "R") {
                        std::cout<<"continuous_angry_or_sad="<<continuous_angry_or_sad<<" | Write RED to serial port"<<std::endl;
                        s = "R";
                        write(serial_port, s, 1);
                    }
                    continuous_angry_or_sad = 0;
                }
                if (continuous_happy == 10) {
                    if (s != "G") {
                        std::cout<<"continuous_happy="<<continuous_happy<<" | Write GREEN to serial port"<<std::endl;
                        s = "G";
                        write(serial_port, s, 1);
                    }
                    continuous_happy = 0;
                }

                if (fr == 10) {
                    // continuous_angry_or_sad = 0;
                    std::cout<<"last_point reached"<<std::endl;
                    if (s != "G" && emo != 4 && emo != 2) {
                        std::cout<<"emo="<<emo<<" || Write GREEN to serial port"<<std::endl;
                        s = "G";
                        write(serial_port, s, 1);
                    }

                    break;
                }
            }
        }

    }
}

} // anonymous namespace

void DrawPolyline(const std::vector<cv::Point> &polyline,
                  const cv::Scalar &color, cv::Mat *image, int lwd)
{
    PT_CHECK(image);
    PT_CHECK(!image->empty());
    PT_CHECK_EQ(image->type(), CV_8UC3);
    PT_CHECK_GT(lwd, 0);
    PT_CHECK_LT(lwd, 20);

    for (size_t i = 1; i < polyline.size(); i++)
    {
        cv::line(*image, polyline[i - 1], polyline[i], color, lwd);
    }
}

void SaveDetectionLogToTrajFile(const std::string &path,
                                const DetectionLog &log)
{
    std::ofstream file(path.c_str());
    PT_CHECK(file.is_open());
    SaveDetectionLogToStream(file, log);
}

void SaveDetectionLogToTrajFile(const std::string &path,
                                const DetectionLog &log,
                                std::vector<LogObject> &logObjs,
                                const double fps,
                                const int serial_port)
{
    std::ofstream file(path.c_str());
    PT_CHECK(file.is_open());
    SaveDetectionLogToStream(file, log, logObjs, fps, serial_port);
}

std::map<std::string, InferencePlugin>
LoadPluginForDevices(const std::vector<std::string> &devices,
                     const std::string &custom_cpu_library,
                     const std::string &custom_cldnn_kernels,
                     bool should_use_perf_counter)
{
    std::map<std::string, InferencePlugin> plugins_for_devices;

    for (const auto &device : devices)
    {
        if (plugins_for_devices.find(device) != plugins_for_devices.end())
        {
            continue;
        }
        std::cout << "Loading plugin " << device << std::endl;
        InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
        printPluginVersion(plugin, std::cout);
        /** Load extensions for the CPU plugin **/
        if ((device.find("CPU") != std::string::npos))
        {
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
            if (!custom_cpu_library.empty())
            {
                // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                auto extension_ptr = make_so_pointer<IExtension>(custom_cpu_library);
                plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
            }
        }
        else if (!custom_cldnn_kernels.empty())
        {
            // Load Extensions for other plugins not CPU
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE,
                               custom_cldnn_kernels}});
        }
        if (device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos)
        {
            plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
        }
        if (should_use_perf_counter)
            plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
        plugins_for_devices[device] = plugin;
    }
    return plugins_for_devices;
}
