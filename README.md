# Face recognition and tracking C++ Demo

For more information about the pre-trained models, refer to the [Open Model Zoo](https://github.com/opencv/open_model_zoo/tree/master/intel_models/index.md) repository on GitHub*.

## Prerequisites
- Opencv 4.0.0
- Openvino 2019R2
- Python 3

## Download
* [Pre-Trained Models](intel_models/index.md)
* [Public Models Downloader](model_downloader/README.md)
* [Demos](face_recognition/README.md) that demonstrate models usage with Deep Learning Deployment Toolkit


## Creating a Gallery for Face Recognition

To recognize faces on a frame, the demo needs a gallery of reference images. Each image should contain a tight crop of face. You can create the gallery from an arbitrary list of images:
1. Put images containing tight crops of frontal-oriented faces to a separate empty folder. Each identity could have multiple images. Name images as `id_name.0.png, id_name.1.png, ...`.
2. Run the `create_list.py <path_to_folder_with_images>` command to get a list of files and identities in `.json` format.

## Build
Modify `build_dir` and `build_type` in `build_demos.sh` bash script and run.

## Running

Running the application with the `-h` option yields the following usage message:
```
cd $build_dir
```
```sh
./face_reg_track -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

face_reg_track [OPTION]
Options:

    -h                           Print a usage message.
    -i "<path>"                  Required. Path to a video file or a folder with images (all images should have names 0000000001.jpg, 0000000002.jpg, etc).
    -m_det "<path>"              Required. Path to the Pedestrian Detection Retail model (.xml) file.
    -m_reid "<path>"             Required. Path to the Pedestrian Reidentification Retail model (.xml) file.
    -l "<absolute_path>"         Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
    -c "<absolute_path>"         Optional. For GPU custom kernels, if any. Absolute path to the .xml file with the kernels description.
    -d_det "<device>"            Optional. Specify the target device for pedestrian detection (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). 
    -d_reid "<device>"           Optional. Specify the target device for pedestrian reidentification (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). 
    -pc                          Optional. Enable per-layer performance statistics.
    -crop_gallery                Optional. Crop images during faces gallery creation.
    -out "<path>"                Optional. The file name to write output log file with results of pedestrian tracking. The format of the log file is compatible with MOTChallenge format.
```

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

Example of a valid command line to run the application with pre-trained models for recognizing identities from faces_gallery:
```
cd $build_dir
sh ./face_reg_track \
 -m_fd <path_to_model>/face-detection-retail-0004.xml \
 -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
 -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
 -fg <path_to_faces_gallery.json> \
 -d_fd GPU -d_reid CPU -d_lm CPU \
 -t_fd 0.7 -t_reid 0.6 \
 -i <path_to_video_sources> \
 -out <path_to_output_folder>
```


```
face_reg_track_debug \
 -m_det $mFDR32  -m_reid $mPRI32  -m_lm $mLMR32  -m_hp $mHP32 \
 -fg $facesGal \
 -d_det CPU  -d_reid CPU  -d_lm CPU  -d_hp CPU \
 -t_det 0.5  -t_reid 0.5 \
 -min_size_fr 128 -crop_gallery -auto_reg -debug \
 -i /dev/video0 \
 -out out
```

```
face_reg_track_debug \
 -m_det $mFDR32  -m_reid $mPRI32  -m_lm $mLMR32  -m_hp $mHP32 \
 -fg $facesGal \
 -d_det CPU  -d_reid CPU  -d_lm CPU  -d_hp CPU \
 -t_det 0.7  -t_reid 0.3 \
 -min_size_fr 128 \
 -i rtsp://admin:admin123@192.168.1.108:554 \
 -out out/08.31_21h15
```

```
face_reg_track_debug \
 -m_det $mFDR32  -m_reid $mPRI32  -m_lm $mLMR32  -m_hp $mHP32 \
 -fg ugly_template.json \
 -d_det CPU  -d_reid CPU  -d_lm CPU  -d_hp CPU \
 -t_det 0.7  -t_reid 0.3 \
 -min_size_fr 128 \
 -i rtsp://admin:admin123@192.168.1.108:554 \
 -out out/08.31_21h15
```