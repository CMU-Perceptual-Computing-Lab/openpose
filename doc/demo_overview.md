OpenPose Demo - Overview
====================================

Forget about the OpenPose library code, just compile the library and use the demo `./build/examples/openpose/openpose.bin`.

In order to learn how to use it, run `./build/examples/openpose/openpose.bin --help` in your bash and read all the available flags (check only the flags for `examples/openpose/openpose.cpp` itself, i.e. the section `Flags from examples/openpose/openpose.cpp:`). We detail some of them in the following sections.



## Other Important Options
Please, in order to check all the real time pose demo options and their details, run `./build/examples/openpose/openpose.bin --help`. We describe here some of the most important ones.

`--video input.mp4`: Input video. If omitted, it will use the webcam.

`--camera 3`: Choose webcam number (default: 0). If `--camera`, `--image_dir` and `--write_video` are omitted, it is equivalent to use `--camera 0`.

`--image_dir path_to_images/`: Run on all images (jpg, png, bmp, etc.) in `path_to_images/`. You can test the program with the image directory `examples/media/`.

`--write_video path.avi`: Render images with this prefix: `path.avi`. You can test the program with the example video `examples/media/video.avi`.

`--write_pose path/`: Output JSON, XML or YML files with the people pose data on the `path/` folder.

`--process_real_time`: It might skip frames in order to keep the final output displaying frames on real time.

`--disable_blending`: If selected, it will only render the pose skeleton or desired heat maps, while blocking the original background. Also related: `part_to_show`, `alpha_pose`, and `alpha_pose`.

`--part_to_show`: Select the prediction channel to visualize (default: 0). 0 to visualize all the body parts, 1-18 for each body part heat map, 19 for the background heat map, 20 for all the body part heat maps together, 21 for all the PAFs, 22-69 for each body part pair PAF.

`--no_display`: Display window not opened. Useful if there is no X server and/or to slightly speed up the processing if visual output is not required.

`--num_gpu 2 --num_gpu_start 0`: Parallelize over this number of GPUs starting by the desired device id. Default is 1 and 0, respectively.

`--num_scales 3 --scale_gap 0.15`: Use 3 scales, 1, (1-0.15), (1-0.15*2). Default is one scale. If you want to change the initial scale, you actually want to multiply your desired initial scale by the `net_resolution`.

`--net_resolution 656x368 --resolution 1280x720`: For HD images and video (default values).

`--net_resolution 496x368 --resolution 640x480`: For VGA images and video.

`--model_pose MPI`: It will use MPI (15 body keypoints). Default: COCO (18 body keypoints). MPI is slightly faster. The variation `MPI_4_layers` sacrifies accuracy in order to further increase speed.

`--logging_level 3`: Logging messages threshold, range [0,255]: 0 will output any message & 255 will output none. Current messages in the range [1-4], 1 for low priority messages and 4 for important ones.

## Multiple Scales
Running at multiple scales might drastically slow down the speed, but it will increase the accuracy. Given the CNN input size (set with `net_resolution`), `num_scales` and `scale_gap` configure the number of scales to use and the gap between them, respectively. For instance, `--num_scales 3 --scale_gap 0.15` means using 3 scales at resolution: (1), (1-0.15) and (1-2*0.15) times the `net_resolution`.

## Heat Maps Storing
The following command will save all the body part heat maps, background heat map and Part Affinity Fields (PAFs) in the folder `output_heatmaps_folder`. It will save them on PNG format. Instead of individually saving each of the 67 heatmaps (18 body parts + background + 2 x 19 PAFs) individually, the library concatenate them vertically into a huge (width x #heatmaps) x (height) matrix. The PAFs channels are multiplied by 2 because there is one heatmpa for the x-coordinates and one for the y-coordinates. The order is body parts + bkg + PAFs. It will follow the sequence on POSE_BODY_PART_MAPPING in [include/openpose/pose/poseParameters.hpp](../include/openpose/pose/poseParameters.hpp).
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --heatmaps_add_parts --heatmaps_add_bkg --heatmaps_add_PAFs --write_heatmaps output_heatmaps_folder/
```



## Example
The following example runs the video `vid.mp4`, renders image frames on `output/result.avi`, and outputs JSON files as `output/%12d.json`, parallelizing over 2 GPUs:
```
./build/examples/openpose/openpose.bin --video examples/media/video.avi --num_gpu 2 --write_video output/result.avi --write_json output/
```
