OpenPose Demo - Overview
====================================

Forget about the OpenPose library code, just compile the library and use the demo `./build/examples/openpose/openpose.bin`.

In order to learn how to use it, run `./build/examples/openpose/openpose.bin --help` in your bash and read all the available flags (check only the flags for `examples/openpose/openpose.cpp` itself, i.e. the section `Flags from examples/openpose/openpose.cpp:`). We detail some of them in the following sections.



## Running on Images, Video or Webcam
See [doc/quick_start.md#quick-start](./quick_start.md#quick-start).



## Pose + Face + Hands
See [doc/quick_start.md#quick-start](./quick_start.md#quick-start).



## Maximum Accuracy Configuration
See [doc/quick_start.md#quick-start](./quick_start.md#quick-start).



## Kinect 2.0 as Webcam on Windows 10
Since the Windows 10 Anniversary, Kinect 2.0 can be read as a normal webcam. All you need to do is go to `device manager`, expand the `kinect sensor devices` tab, right click and update driver of `WDF kinectSensor Interface`. If you already have another webcam, disconnect it or use `--camera 2`.



## JSON Output with No Visualization
The following example runs the demo video `video.avi` and outputs JSON files in `output/`. Note: see [doc/output.md](./output.md) to understand the format of the JSON files.
```
# Only body
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_keypoint_json output/ --no_display --render_pose 0
# Body + face + hands
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_keypoint_json output/ --no_display --render_pose 0 --face --hand
```



## JSON Output + Rendered Images Saving
The following example runs the demo video `video.avi`, renders image frames on `output/result.avi`, and outputs JSON files in `output/`. Note: see [doc/output.md](./output.md) to understand the format of the JSON files.
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_video output/result.avi --write_keypoint_json output/
```



## Hands
```
# Fast method for speed
./build/examples/openpose/openpose.bin --hand
# Best results found with 6 scales
./build/examples/openpose/openpose.bin --hand --hand_scale_number 6 --hand_scale_range 0.4
# Adding tracking to Webcam (if FPS per GPU > 10 FPS) and Video
./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_tracking
# Multi-scale + tracking is also possible
./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_scale_number 6 --hand_scale_range 0.4 --hand_tracking
```



## Rendering Face and Hands without Pose
```
# CPU rendering (faster)
./build/examples/openpose/openpose.bin --render_pose 0 --face --face_render 1 --hand --hand_render 1
# GPU rendering
./build/examples/openpose/openpose.bin --render_pose 0 --face --face_render 2 --hand --hand_render 2
```



## Debugging Information
```
# Basic information
./build/examples/openpose/openpose.bin --logging_level 3
# Showing all messages
./build/examples/openpose/openpose.bin --logging_level 0
```



## Selecting Some GPUs
The following example runs the demo video `video.avi`, parallelizes it over 2 GPUs, GPUs 1 and 2 (note that it will skip GPU 0):
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --num_gpu 2 --num_gpu_start 1
```



## Heat Maps Storing
The following command will save all the body part heat maps, background heat map and Part Affinity Fields (PAFs) in the folder `output_heatmaps_folder`. It will save them on PNG format. Instead of individually saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenate them vertically into a huge (width x #heatmaps) x (height) matrix. The PAFs channels are multiplied by 2 because there is one heatmpa for the x-coordinates and one for the y-coordinates. The order is body parts + bkg + PAFs. It will follow the sequence on POSE_BODY_PART_MAPPING in [include/openpose/pose/poseParameters.hpp](../include/openpose/pose/poseParameters.hpp).
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --heatmaps_add_parts --heatmaps_add_bkg --heatmaps_add_PAFs --write_heatmaps output_heatmaps_folder/
```



## Main Flags
We enumerate some of the most important flags, check the `Flags Detailed Description` section or run `./build/examples/openpose/openpose.bin --help` for a full description of all of them.

- `--face`: Enables face keypoint detection.
- `--hand`: Enables hand keypoint detection.
- `--video input.mp4`: Read video.
- `--camera 3`: Read webcam number 3.
- `--image_dir path_to_images/`: Run on a folder with images.
- `--ip_camera http://iris.not.iac.es/axis-cgi/mjpg/video.cgi?resolution=320x240?x.mjpeg`: Run on a streamed IP camera. See examples public IP cameras [here](http://www.webcamxp.com/publicipcams.aspx).
- `--write_video path.avi`: Save processed images as video.
- `--write_images folder_path`: Save processed images on a folder.
- `--write_keypoint path/`: Output JSON, XML or YML files with the people pose data on a folder.
- `--process_real_time`: For video, it might skip frames to display at real time.
- `--disable_blending`: If enabled, it will render the results (keypoint skeletons or heatmaps) on a black background, not showing the original image. Related: `part_to_show`, `alpha_pose`, and `alpha_pose`.
- `--part_to_show`: Prediction channel to visualize.
- `--no_display`: Display window not opened. Useful for servers and/or to slightly speed up OpenPose.
- `--num_gpu 2 --num_gpu_start 1`: Parallelize over this number of GPUs starting by the desired device id. By default it uses all the available GPUs.
- `--model_pose MPI`: Model to use, affects number keypoints, speed and accuracy.
- `--logging_level 3`: Logging messages threshold, range [0,255]: 0 will output any message & 255 will output none. Current messages in the range [1-4], 1 for low priority messages and 4 for important ones.



## Flags Description
Each flag is divided into flag name, default value, and description.

1. Debugging/Other
- DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
- DEFINE_bool(disable_multi_thread,       false,          "It would slightly reduce the frame rate in order to highly reduce the lag. Mainly useful for 1) Cases where it is needed a low latency (e.g. webcam in real-time scenarios with low-range GPU devices); and 2) Debugging OpenPose when it is crashing to locate the error.");

2. Producer
- DEFINE_int32(camera,                    -1,             "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative number (by default), to auto-detect and open the first available camera.");
- DEFINE_string(camera_resolution,        "1280x720",     "Size of the camera frames to ask for.");
- DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the minimum value between the OpenPose displayed speed and the webcam real frame rate.");
- DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default example video.");
- DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20 images. Read all standard formats (jpg, png, bmp, etc.).");
- DEFINE_string(ip_camera,                "",             "String with the IP camera URL. It supports protocols like RTSP and HTTP.");
- DEFINE_uint64(frame_first,              0,              "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
- DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to 10, it will process 11 frames (0-10).");
- DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
- DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
- DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
- DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is too long, it will skip frames. If it is too fast, it will slow it down.");

3. OpenPose
- DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
- DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the input image resolution.");
- DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your machine.");
- DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
- DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y) coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags. Select `0` to scale it to the original source resolution, `1`to scale it to the net output size (set with `net_resolution`), `2` to scale it to the final output size (set with `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related with `scale_number` and `scale_gap`.");

4. OpenPose Body Pose
- DEFINE_bool(body_disable,               false,          "Disable body keypoint detection. Option only possible for faster (but less accurate) face keypoint detection.");
- DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), `MPI_4_layers` (15 keypoints, even faster but less accurate).");
- DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased, the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect ratio possible to the images or videos to be processed. Using `-1` in any of the dimensions, OP will choose the optimal resolution depending on the other value introduced by the user. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
- DEFINE_int32(scale_number,              1,              "Number of scales to average.");
- DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1. If you want to change the initial scale, you actually want to multiply the `net_resolution` by your desired initial scale.");
- DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array, and analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps (program speed will decrease). Not required for our library, enable it only if you intend to process this information later. If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential memory order: body parts + bkg + PAFs. It will follow the order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
- DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to background.");
- DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");

5. OpenPose Face
- DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g. `model_folder`. Note that this will considerable slow down the performance and increse the required GPU memory. In addition, the greater number of people on the image, the slower OpenPose will be.");
- DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint detector. 320x320 usually works fine while giving a substantial speed up when multiple faces on the image.");

6. OpenPose Hand
- DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g. `model_folder`. Analogously to `--face`, it will also slow down the performance, increase the required GPU memory and its speed depends on the number of people.");
- DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint detector.");
- DEFINE_int32(hand_scale_number,         1,              "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4");
- DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
- DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it simply looks for hands in positions at which hands were located in previous frames, but it does not guarantee the same person ID among frames");

7. OpenPose Rendering
- DEFINE_int32(part_to_show,              0,              "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body part heat map, 19 for the background heat map, 20 for all the body part heat maps together, 21 for all the PAFs, 22-40 for each body part pair PAF");
- DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black background, instead of being rendered into the original image. Related: `part_to_show`, `alpha_pose`, and `alpha_pose`.");

8. OpenPose Rendering Pose
- DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be rendered. Generally, a high threshold (> 0.5) will only render very clear body parts; while small thresholds (~0.1) will also output guessed and occluded keypoints, but also more false positives (i.e. wrong detections).");
- DEFINE_int32(render_pose,               2,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will render both `outputData` and `cvOutputData` with the original image and desired body part to be shown (i.e. keypoints, heat maps or PAFs).");
- DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it. Only valid for GPU rendering.");
- DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the heatmap, 0 will only show the frame. Only valid for GPU rendering.");

9. OpenPose Rendering Face
- DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the face keypoints.");
- DEFINE_int32(face_render,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same configuration that `render_pose` is using.");
- DEFINE_double(face_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to face.");
- DEFINE_double(face_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");

10. OpenPose Rendering Hand
- DEFINE_double(hand_render_threshold,    0.2,            "Analogous to `render_threshold`, but applied to the hand keypoints.");
- DEFINE_int32(hand_render,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same configuration that `render_pose` is using.");
- DEFINE_double(hand_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to hand.");
- DEFINE_double(hand_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");

11. Display
- DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
- DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g. number of current frame and people). It does not affect the pose rendering.");
- DEFINE_bool(no_display,                 false,          "Do not open a display window. Useful if there is no X server and/or to slightly speed up the processing if visual output is not required.");
12. Result Saving
- DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
- DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV function cv::imwrite for all compatible extensions.");
- DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
- DEFINE_string(write_keypoint,           "",             "Directory to write the people body pose keypoint data. Set format with `write_keypoint_format`.");
- DEFINE_string(write_keypoint_format,    "yml",          "File extension and format for `write_keypoint`: json, xml, yaml & yml. Json not available for OpenCV < 3.0, use `write_keypoint_json` instead.");
- DEFINE_string(write_keypoint_json,      "",             "Directory to write people pose data in *.json format, compatible with any OpenCV version.");
- DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
- DEFINE_string(write_heatmaps,           "",             "Directory to write body pose heatmaps in *.png format. At least 1 `add_heatmaps_X` flag must be enabled.");
- DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`. Recommended `png` or any compressed and lossless format.");
