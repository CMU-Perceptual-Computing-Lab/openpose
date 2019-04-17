# Basic commands
## Installing it (1-time thing)
```
mkdir build && cd build && cmake-gui ..
```

## Compiling it
```
cd build
make -j`nproc`
```

## Running it
### Without GUI
```
./build/examples/openpose/openpose.bin --num_gpu 1 --profile_speed 100 --display 0 --video examples/media/video.avi
```

### With GUI
```
./build/examples/openpose/openpose.bin --num_gpu 1 --profile_speed 100 --video examples/media/video.avi
```

### For NVprof
```
# Super fast
./build/examples/openpose/openpose.bin --num_gpu 1 --frame_last 1 --net_resolution -1x128 --display 0 --profile_speed 100 --video examples/media/video.avi
# Fast
./build/examples/openpose/openpose.bin --num_gpu 1 --frame_last 5 --display 0 --profile_speed 100 --video examples/media/video.avi
```




### Testing resize (BODY_135)
clear && clear && cd build/ && make -j`nproc` && cd .. && ./build/examples/openpose/openpose.bin --video soccer.mp4 --profile_speed 100 --num_gpu_start 1 --model_pose BODY_135 --net_resolution -1x480






### Notes for final report:
- 1. CUDA test: All optimized code in joker repo. Compare speed with original one in official OP.
	- Image resize:
		- Reduce #kernels launch: +10%
		- Shared memory:
		- Only write once to memory at the end:
		- Multi-scale? (e.g. write once + single kernel)
- 2. AVX: Enable/disable AVX flag.
