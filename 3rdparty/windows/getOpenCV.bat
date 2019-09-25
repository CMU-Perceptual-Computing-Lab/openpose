:: Avoid printing all the comments in the Windows cmd
@echo off

SET UNZIP_EXE=unzip\unzip.exe
SET WGET_EXE=wget\wget.exe

:: Download temporary zip
echo ----- Downloading OpenCV -----
SET OPENCV_FOLDER=opencv\
SET ZIP_NAME=opencv_411_v14_15_2019_09_24.zip
SET ZIP_FULL_PATH=%OPENCV_FOLDER%%ZIP_NAME%
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/%ZIP_NAME% -P %OPENCV_FOLDER%
echo:

echo ----- Unzipping OpenCV -----
%UNZIP_EXE% %ZIP_FULL_PATH%
echo:

echo ----- Deleting Temporary Zip File %ZIP_FULL_PATH% -----
del "%ZIP_FULL_PATH%"

echo ----- OpenCV Downloaded and Unzipped -----
