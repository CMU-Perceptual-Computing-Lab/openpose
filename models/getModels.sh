# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"
FACE_FOLDER="face/"
HAND_FOLDER="hand/"

# ------------------------- POSE MODELS -------------------------
# Body (COCO)
COCO_FOLDER=${POSE_FOLDER}"coco/"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}
# Alternative: it will not check whether file was fully downloaded
# if [ ! -f $COCO_MODEL ]; then
#     wget ${OPENPOSE_URL}$COCO_MODEL -P $COCO_FOLDER
# fi

# Body (MPI)
MPI_FOLDER=${POSE_FOLDER}"mpi/"
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}

# "------------------------- FACE MODELS -------------------------"
# Face
FACE_MODEL=${FACE_FOLDER}"pose_iter_116000.caffemodel"
wget -c ${OPENPOSE_URL}${FACE_MODEL} -P ${FACE_FOLDER}

# "------------------------- HAND MODELS -------------------------"
# Hand
HAND_MODEL=$HAND_FOLDER"pose_iter_102000.caffemodel"
wget -c ${OPENPOSE_URL}${HAND_MODEL} -P ${HAND_FOLDER}
