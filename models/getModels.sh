# "------------------------- POSE MODELS -------------------------"
# Downloading body pose (COCO and MPI) as well as face models
pose_folder="pose/"
face_folder="face/"

# Body (COCO)
coco_folder="$pose_folder"coco/""
coco_model="$coco_folder"pose_iter_440000.caffemodel""
wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/$coco_model -P $coco_folder

# Body (MPI)
mpi_folder="$pose_folder"mpi/""
mpi_model="$mpi_folder"pose_iter_160000.caffemodel""
wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/$mpi_model -P $mpi_folder

# Face
face_model="$face_folder"pose_iter_116000.caffemodel""
wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/$face_model -P $face_folder

