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
./build/examples/openpose/openpose.bin --num_gpu 1 --frame_last 5 --display 0 --profile_speed 100 --video examples/media/video.avi
```
