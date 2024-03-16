# ------------------------- Downlaod MODELS -------------------------
# (ISSUE #1602) Maintainer has provided alternative model link with google drive.
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411
MODEL_URL="https://drive.google.com/uc?id=1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh"
FILENAME='models.zip'

# Supplementary package 'gdown' to download google drive
if ! [ -x "$(command -v gdown)" ]; then
    echo "Please install gdown package with pip"
    exit
fi

cd ..
gdown ${MODEL_URL} -O ${FILENAME}
unzip ${FILENAME}
cd -

RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "Model download success"
  exit
else
  echo "Model download failure"
  echo "Fallback to original approach"
fi

# ------------------------- BODY, FOOT, FACE, AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"
FACE_FOLDER="face/"
HAND_FOLDER="hand/"

# ------------------------- POSE (BODY+FOOT) MODELS -------------------------
# Body (BODY_25)
BODY_25_FOLDER=${POSE_FOLDER}"body_25/"
BODY_25_MODEL=${BODY_25_FOLDER}"pose_iter_584000.caffemodel"
wget -c ${OPENPOSE_URL}${BODY_25_MODEL} -P ${BODY_25_FOLDER}

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
