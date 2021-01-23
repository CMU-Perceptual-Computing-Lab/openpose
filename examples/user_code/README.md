Adding and Testing Custom Code
====================================



## Purpose
You can quickly add your custom code into this folder so that quick prototypes can be easily tested without having to create a whole new project just for it.



## How-to
1. Install/compile OpenPose as usual.
2. Add your custom cpp / hpp files in this folder. Hint: You might want to start by copying the [OpenPoseDemo](../openpose/openpose.cpp) example or any of the [examples/tutorial_api_cpp/](../tutorial_api_cpp/) examples.
3. Add the name of your custom cpp / hpp files at the top of the [examples/user_code/CMakeLists.txt](CMakeLists.txt) file.
4. Re-run CMake (`Configure` and `Generate`).
5. Then, you can simply modify their content.
6. You can know compile OpenPose after each change in your files:
	- Ubuntu:
	```
	cd build/
	make -j`nproc`
	```
	- Mac:
	```
	cd build/
	make -j`sysctl -n hw.logicalcpu`
	```
	- Windows: Compile from Visual Studio (F5, F7, green play button, etc.).
5. **Run step 4 every time that you make changes into your code**.



## Running your Custom Code
Run:
```
./build/examples/user_code/{your_custom_file_name}
```
