:: ------------------------- BODY, FACE AND HAND MODELS -------------------------

:: Avoid printing all the comments in the Windows cmd
@echo off

echo:
echo:
echo:

echo ----- Downloading body pose (COCO and MPI), face and hand models -----
SET pose_folder=pose/
SET face_folder=face/
SET wgetExe=..\3rdparty\windows\wget\wget.exe

echo:
echo ------------------------- POSE MODELS -------------------------
echo Body (COCO)
SET coco_folder=%pose_folder%coco/
SET coco_model=%coco_folder%pose_iter_440000.caffemodel
%wgetExe% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%coco_model% -P %coco_folder%

echo:
echo Body (MPI)
SET mpi_folder=%pose_folder%mpi/
SET mpi_model=%mpi_folder%pose_iter_160000.caffemodel
%wgetExe% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%mpi_model% -P %mpi_folder%
echo ----------------------- POSE DOWNLOADED -----------------------

echo:
echo ------------------------- FACE MODELS -------------------------
echo Face
SET face_model=%face_folder%pose_iter_116000.caffemodel
%wgetExe% -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/%face_model% -P %face_folder%
echo ----------------------- FACE DOWNLOADED -----------------------
