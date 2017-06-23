OpenPose Demo - Overview
====================================

Forget about the OpenPose library code, just compile the library and use the demo `./build/examples/openpose/openpose.bin`.

In order to learn how to use it, run `./build/examples/openpose/openpose.bin --help` in your bash and read all the available flags (check only the flags for `examples/openpose/openpose.cpp` itself, i.e. the section `Flags from examples/openpose/openpose.cpp:`). We detail some of them in the following sections.



## All Flags
Each flag is divided into flag name, default value, and description.

1. Debugging
- DEFINE_int32(logging_level,             4,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
2. Producer
- DEFINE_int32(camera,                    0,              "The camera index for cv::VideoCapture. Integer in the range [0, 9].");
- DEFINE_string(camera_resolution,        "1280x720",     "Size of the camera frames to ask for.");
- DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the minimum value between the OpenPose displayed speed and the webcam real frame rate.");
- DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default example video.");
- DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20 images.");
- DEFINE_uint64(frame_first,              0,              "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
- DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to 10, it will process 11 frames (0-10).");
- DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
- DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
- DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
3. OpenPose
- DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
- DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the default images resolution.");
- DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your machine.");
- DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
- DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y) coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags. Select `0` to scale it to the original source resolution, `1`to scale it to the net output size (set with `net_resolution`), `2` to scale it to the final output size (set with `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related with `num_scales` and `scale_gap`.");
4. OpenPose Body Pose
- DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
- DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased, the speed increases.");
- DEFINE_int32(num_scales,                1,              "Number of scales to average.");
- DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you want to change the initial scale, you actually want to multiply the `net_resolution` by your desired initial scale.");
- DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array (program speed will decrease). Not required for our library, enable it only if you intend to process this information later. If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential memory order: body parts + bkg + PAFs. It will follow the order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
- DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to background.");
- DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
5. OpenPose Face
- DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g. `model_folder`.");
- DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16. Analogous to `net_resolution` but applied to the face keypoint detector. 320x320 usually works fine while giving a substantial speed up when multiple faces on the image.");
6. OpenPose Hand
- DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g. `model_folder`.");
- DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16. Analogous to `net_resolution` but applied to the hand keypoint detector. 320x320 usually works fine while giving a substantial speed up when multiple hands on the image.");t_resolution` but applied to the hand keypoint detector.");
- DEFINE_int32(hand_detection_mode,       -1,             "Set to 0 to perform 1-time keypoint detection (fastest), 1 for iterative detection (recommended for images and fast videos, slow method), 2 for tracking (recommended for webcam if the frame rate is >10 FPS per GPU used and for video, in practice as fast as 1-time detection), 3 for both iterative and tracking (recommended for webcam if the resulting frame rate is still >10 FPS and for video, ideally best result but slower), or -1 (default) for automatic selection (fast method for webcam, tracking for video and iterative for images).");
7. OpenPose Rendering
- DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
- DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it will only display the results.");
8. OpenPose Rendering Pose
- DEFINE_int32(render_pose,               2,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will render both `outputData` and `cvOutputData` with the original image and desired body part to be shown (i.e. keypoints, heat maps or PAFs).");
- DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it. Only valid for GPU rendering.");
- DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the heatmap, 0 will only show the frame. Only valid for GPU rendering.");
9. OpenPose Rendering Face
- DEFINE_int32(render_face,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same configuration that `render_pose` is using.");
- DEFINE_double(alpha_face,               0.6,            "Analogous to `alpha_pose` but applied to face.");
- DEFINE_double(alpha_heatmap_face,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
10. OpenPose Rendering Hand
- DEFINE_int32(render_hand,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same configuration that `render_pose` is using.");
- DEFINE_double(alpha_hand,               0.6,            "Analogous to `alpha_pose` but applied to hand.");
- DEFINE_double(alpha_heatmap_hand,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
11. Display
- DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
- DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is too long, it will skip frames. If it is too fast, it will slow it down.");
- DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g. number of current frame and people). It does not affect the pose rendering.");
- DEFINE_bool(no_display,                 false,          "Do not open a display window.");
12. Result Saving
- DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
- DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV function cv::imwrite for all compatible extensions.");
- DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
- DEFINE_string(write_keypoint,           "",             "Directory to write the people body pose keypoint data. Set format with `write_keypoint_format`.");
- DEFINE_string(write_keypoint_format,    "yml",          "File extension and format for `write_keypoint`: json, xml, yaml & yml. Json not available for OpenCV < 3.0, use `write_keypoint_json` instead.");
- DEFINE_string(write_keypoint_json,      "",             "Directory to write people pose data in *.json format, compatible with any OpenCV version.");
- DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
- DEFINE_string(write_heatmaps,           "",             "Directory to write heatmaps in *.png format. At least 1 `add_heatmaps_X` flag must be enabled.");
- DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`. Recommended `png` or any compressed and lossless format.");

## Multiple Scales
Running at multiple scales might drastically slow down the speed, but it will increase the accuracy. Given the CNN input size (set with `net_resolution`), `num_scales` and `scale_gap` configure the number of scales to use and the gap between them, respectively. For instance, `--num_scales 3 --scale_gap 0.15` means using 3 scales at resolution: (1), (1-0.15) and (1-2*0.15) times the `net_resolution`.

## Heat Maps Storing
The following command will save all the body part heat maps, background heat map and Part Affinity Fields (PAFs) in the folder `output_heatmaps_folder`. It will save them on PNG format. Instead of individually saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenate them vertically into a huge (width x #heatmaps) x (height) matrix. The PAFs channels are multiplied by 2 because there is one heatmpa for the x-coordinates and one for the y-coordinates. The order is body parts + bkg + PAFs. It will follow the sequence on POSE_BODY_PART_MAPPING in [include/openpose/pose/poseParameters.hpp](../include/openpose/pose/poseParameters.hpp).
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --heatmaps_add_parts --heatmaps_add_bkg --heatmaps_add_PAFs --write_heatmaps output_heatmaps_folder/
```



## Some Important Configuration Flags
Please, in order to check all the real time pose demo options and their details, run `./build/examples/openpose/openpose.bin --help`. We describe here some of the most important ones.

- `--face`: If enabled, it will also detect the faces on the image. Note that this will considerable slow down the performance and increse the required GPU memory. In addition, the greater number of people on the image, the slower OpenPose will be.
- `--hand`: Analogously to `--face`, but applied to hands. Note that this will also slow down the performance, increse the required GPU memory and its speed depends on the number of people.
- `--video input.mp4`: Input video. If omitted, it will use the webcam.
- `--camera 3`: Choose webcam number (default: 0). If `--camera`, `--image_dir` and `--write_video` are omitted, it is equivalent to use `--camera 0`.
- `--image_dir path_to_images/`: Run on all images (jpg, png, bmp, etc.) in `path_to_images/`. You can test the program with the image directory `examples/media/`.
- `--write_video path.avi`: Render images with this prefix: `path.avi`. You can test the program with the example video `examples/media/video.avi`.
- `--write_images folder_path`: Render images on the folder `folder_path`.
- `--write_keypoint path/`: Output JSON, XML or YML files with the people pose data on the `path/` folder.
- `--process_real_time`: It might skip frames in order to keep the final output displaying frames on real time.
- `--disable_blending`: If selected, it will only render the pose skeleton or desired heat maps, while blocking the original background. Also related: `part_to_show`, `alpha_pose`, and `alpha_pose`.
- `--part_to_show`: Select the prediction channel to visualize (default: 0). 0 to visualize all the body parts, 1-18 for each body part heat map, 19 for the background heat map, 20 for all the body part heat maps together, 21 for all the PAFs, 22-69 for each body part pair PAF.
- `--no_display`: Display window not opened. Useful if there is no X server and/or to slightly speed up the processing if visual output is not required.
- `--num_gpu 2 --num_gpu_start 1`: Parallelize over this number of GPUs starting by the desired device id. Default `num_gpu` is -1, which will use all the available GPUs.
- `--num_scales 3 --scale_gap 0.15`: Use 3 scales, 1, (1-0.15), (1-0.15*2). Default is one scale. If you want to change the initial scale, you actually want to multiply your desired initial scale by the `net_resolution`.
- `--net_resolution 656x368 --resolution 1280x720`: For HD images and video (default values).
- `--net_resolution 496x368 --resolution 640x480`: For VGA images and video.
- `--model_pose MPI`: It will use MPI (15 body keypoints). Default: COCO (18 body keypoints). MPI is slightly faster. The variation `MPI_4_layers` sacrifies accuracy in order to further increase speed.
- `--logging_level 3`: Logging messages threshold, range [0,255]: 0 will output any message & 255 will output none. Current messages in the range [1-4], 1 for low priority messages and 4 for important ones.



## Hands
Very important note, use `hand_detection_mode` accordingly.
```
# Images
# Fast method for speed
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 0
# Iterative for higher accuracy
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 1

# Video
# Iterative tracking for higher accuracy
./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_detection_mode 3
# Tracking for speed
./build/examples/openpose/openpose.bin --video examples/media/video.avi --hand --hand_detection_mode 2

# Webcam
# Fast method for speed if the frame rate is low
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 0
# Iterative for higher accuracy (but the frame rate will be reduced)
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 1
# Tracking for higher accuracy if the frame rate is high enough. Worse results than fast method if frame rate is low
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 2
# Iterative + tracking for best accuracy if frame rate is high enough. Worse results than fast method if frame rate is low
./build/examples/openpose/openpose.bin --hand --hand_detection_mode 3
```



## Debugging Information
```
# Basic information
./build/examples/openpose/openpose.bin --logging_level 3
# Showing all messages
./build/examples/openpose/openpose.bin --logging_level 0
```



## Pose + Face + Hands
```
./build/examples/openpose/openpose.bin --face --hand
```



## Rendering Face without Pose
```
# CPU rendering (faster)
./build/examples/openpose/openpose.bin --face --render_pose 0 --render_face 1
# GPU rendering
./build/examples/openpose/openpose.bin --face --render_pose 0 --render_face 2
```



## Basic Output Saving
The following example runs the demo video `video.avi`, renders image frames on `output/result.avi`, and outputs JSON files in `output/`. It parallelizes over 2 GPUs, GPUs 1 and 2 (note that it will skip GPU 0):
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --num_gpu 2 --num_gpu_start 1 --write_video output/result.avi --write_keypoint_json output/
```
