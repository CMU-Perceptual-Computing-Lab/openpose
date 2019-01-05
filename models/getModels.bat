:: Avoid printing all the comments in the Windows cmd
@echo off

echo ------------------------- BODY, FOOT, FACE, AND HAND MODELS -------------------------
echo ----- Downloading body pose (COCO and MPI), face and hand models -----
SET WGET_EXE=..\3rdparty\windows\wget\wget.exe
SET OPENPOSE_URL=http://posefs1.perception.cs.cmu.edu/OpenPose/models/
SET POSE_FOLDER=pose/
SET FACE_FOLDER=face/
SET HAND_FOLDER=hand/

echo:
echo ------------------------- POSE (BODY+FOOT) MODELS -------------------------
echo Body (BODY_25)
set BODY_25_FOLDER=%POSE_FOLDER%body_25/
set BODY_25_MODEL=%BODY_25_FOLDER%pose_iter_584000.caffemodel
%WGET_EXE% -c %OPENPOSE_URL%%BODY_25_MODEL% -P %BODY_25_FOLDER%

echo Body (COCO)
SET COCO_FOLDER=%POSE_FOLDER%coco/
SET COCO_MODEL=%COCO_FOLDER%pose_iter_440000.caffemodel
%WGET_EXE% -c %OPENPOSE_URL%%COCO_MODEL% -P %COCO_FOLDER%

echo:
echo Body (MPI)
SET MPI_FOLDER=%POSE_FOLDER%mpi/
SET MPI_MODEL=%MPI_FOLDER%pose_iter_160000.caffemodel
%WGET_EXE% -c %OPENPOSE_URL%%MPI_MODEL% -P %MPI_FOLDER%
echo ----------------------- POSE DOWNLOADED -----------------------

echo:
echo ------------------------- FACE MODELS -------------------------
echo Face
SET FACE_MODEL=%FACE_FOLDER%pose_iter_116000.caffemodel
%WGET_EXE% -c %OPENPOSE_URL%%FACE_MODEL% -P %FACE_FOLDER%
echo ----------------------- FACE DOWNLOADED -----------------------

echo:
echo ------------------------- HAND MODELS -------------------------
echo Hand
SET HAND_MODEL=%HAND_FOLDER%pose_iter_102000.caffemodel
%WGET_EXE% -c %OPENPOSE_URL%%HAND_MODEL% -P %HAND_FOLDER%
echo ----------------------- HAND DOWNLOADED -----------------------
