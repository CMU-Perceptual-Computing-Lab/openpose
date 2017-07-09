:: Avoid printing all the comments in the Windows cmd
@echo off

echo ------------------------- BODY, FACE AND HAND MODELS -------------------------
echo ----- Downloading body pose (COCO and MPI), face and hand models -----
SET WGET_EXE=..\3rdparty\windows\wget\wget.exe
SET OPENPOSE_URL=http://posefs1.perception.cs.cmu.edu/OpenPose/models/
SET POSE_FOLDER=pose/
SET FACE_FOLDER=face/
SET HAND_FOLDER=hand/

echo:
echo ------------------------- POSE MODELS -------------------------
echo Body (COCO)
SET COCO_FOLDER=%POSE_FOLDER%coco/
SET COCO_MODEL=%COCO_FOLDER%pose_iter_440000.caffemodel
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%COCO_MODEL% -P %COCO_FOLDER%

echo:
echo Body (MPI)
SET MPI_FOLDER=%POSE_FOLDER%mpi/
SET MPI_MODEL=%MPI_FOLDER%pose_iter_160000.caffemodel
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%MPI_MODEL% -P %MPI_FOLDER%
echo ----------------------- POSE DOWNLOADED -----------------------

echo:
echo ------------------------- FACE MODELS -------------------------
echo Face
SET FACE_MODEL=%FACE_FOLDER%pose_iter_116000.caffemodel
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%FACE_MODEL% -P %FACE_FOLDER%
echo ----------------------- FACE DOWNLOADED -----------------------

echo:
echo ------------------------- HAND MODELS -------------------------
echo Hand
SET HAND_MODEL=%HAND_FOLDER%pose_iter_102000.caffemodel
%WGET_EXE% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%HAND_MODEL% -P %HAND_FOLDER%
echo ----------------------- HAND DOWNLOADED -----------------------
