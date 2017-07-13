:: Avoid printing all the comments in the Windows cmd
@echo off



echo ------------------------- Downloading Caffe and OpenCV -------------------------
echo NOTE: This script assumes that CUDA, cuDNN and Visual Studio are already installed on your machine. Otherwise, it might fail.



:: Go back to main OpenPose folder
cd ..



echo:
echo ------------------------- Download 3rdparty Libraries -------------------------
cd 3rdparty\windows\
echo Downloading Caffe...
call getCaffe.bat

echo Downloading Caffe dependencies...
call getCaffe3rdparty.bat

echo:
echo Downloading OpenCV...
call getOpenCV.bat

:: Go back to OpenPose folder
cd ..\..
echo ------------------------ 3rdparty Libraries Downloaded ------------------------



echo:
echo:
echo:



echo ------------------------- Download Models -------------------------
cd models\
call getModels.bat
:: Go back to OpenPose folder
cd ..
echo ------------------------ Models Downloaded ------------------------



echo:
echo ------------------------- Caffe and OpenCV Downloaded -------------------------

:: Pause until user manually closes it
pause
