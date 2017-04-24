# "------------------------- POSE MODELS -------------------------"
# Downloading COCO and MPI models
models_folder="pose/"

# COCO
coco_folder="$models_folder"coco/""
coco_model="$coco_folder"pose_iter_440000.caffemodel""
if [ ! -f $coco_model ]; then
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/coco/pose_iter_440000.caffemodel -P $coco_folder
fi

# MPI
mpi_folder="$models_folder"mpi/""
mpi_model="$mpi_folder"pose_iter_160000.caffemodel""
if [ ! -f $mpi_model ]; then
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel -P $mpi_folder
fi



# EXTRA - Other paths
# /media/posenas1b/Users/zhe/arch/MPI_exp_caffe/pose43/exp04/model/pose_iter_264000.caffemodel
# cubeserver1:/home/zhe/Real-time-CPM-for-multiple-people/model/pose_iter_166000.caffemodel
