:: Avoid printing all the comments in the Windows cmd
@echo off

SET UNZIP_EXE=unzip\unzip.exe
SET WGET_EXE=wget\wget.exe

:: Download temporary zip
echo ----- Downloading Caffe -----
SET SPINNAKER_FOLDER=spinnaker\
SET ZIP_NAME=spinnaker_2018_01_24.zip
SET ZIP_FULL_PATH=%SPINNAKER_FOLDER%%ZIP_NAME%
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/%ZIP_NAME% -P %SPINNAKER_FOLDER%
echo:

echo ----- Unzipping Caffe -----
%UNZIP_EXE% %ZIP_FULL_PATH%
echo:

:: echo ----- Deleting Temporary Zip File %ZIP_FULL_PATH% -----
:: del "%ZIP_FULL_PATH%"

echo ----- Caffe Downloaded and Unzipped -----
