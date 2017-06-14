@echo off

pushd %~dp0
if exist .\caffe-windows\lib\Release\caffe.lib (
if exist .\caffe-windows\lib\Release\proto.lib (
if exist .\caffe-windows\lib\Debug\caffe-d.lib (
if exist .\caffe-windows\lib\Debug\proto-d.lib (
echo ------------------------- Caffe Already Built -------------------------
exit
)
)
)
)

echo ------------------------- Copying Build Script -------------------------
copy patches\windows\build_win_release.cmd caffe-windows\caffe-windows\scripts
copy patches\windows\build_win_debug.cmd caffe-windows\caffe-windows\scripts

cd caffe-windows\caffe-windows

echo ------------------------- Building Caffe Release Version -------------------------
call .\scripts\build_win_release.cmd
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo ------------------------- Caffe Release Version Built -------------------------
ping 127.1 -n 3 > nul


echo ------------------------- Building Caffe Debug Version -------------------------
call .\scripts\build_win_debug.cmd
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo=
echo ------------------------- Caffe Debug Version Built -------------------------
ping 127.1 -n 3 > nul


cd ..
if NOT EXIST ..\..\windows_project\x64\Debug mkdir ..\..\windows_project\x64\Debug
if NOT EXIST ..\..\windows_project\x64\Release mkdir ..\..\windows_project\x64\Release

echo ------------------------- Copying Debug Version Dependencies -------------------------
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_chrono-vc140-mt-gd-1_61.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_filesystem-vc140-mt-gd-1_61.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_python-vc140-mt-gd-1_61.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_system-vc140-mt-gd-1_61.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_thread-vc140-mt-gd-1_61.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\caffehdf5_D.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\caffehdf5_hl_D.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\caffezlibd1.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\gflagsd.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\glogd.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\libgcc_s_seh-1.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\libgfortran-3.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\libopenblas.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\libquadmath-0.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\opencv_core310d.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_ffmpeg310_64.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_highgui310d.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\opencv_imgcodecs310d.dll ..\..\windows_project\x64\Debug
copy caffe-windows\build\install\bin\opencv_imgproc310d.dll ..\..\windows_project\x64\Debug
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_videoio310d.dll ..\..\windows_project\x64\Debug


echo=
echo=
echo=

echo ------------------------- Copying Release Version Dependencies -------------------------
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_chrono-vc140-mt-1_61.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_filesystem-vc140-mt-1_61.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_python-vc140-mt-1_61.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_system-vc140-mt-1_61.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib\boost_thread-vc140-mt-1_61.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\caffehdf5.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\caffehdf5_hl.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\caffezlib1.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\gflags.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\glog.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\libgcc_s_seh-1.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\libgfortran-3.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\libopenblas.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\libquadmath-0.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\opencv_core310.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_ffmpeg310_64.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_highgui310.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\opencv_imgcodecs310.dll ..\..\windows_project\x64\Release
copy caffe-windows\build\install\bin\opencv_imgproc310.dll ..\..\windows_project\x64\Release
copy dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin\opencv_videoio310.dll ..\..\windows_project\x64\Release

popd