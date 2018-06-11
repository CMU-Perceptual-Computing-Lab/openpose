import numpy as np
import ctypes as ct
import cv2
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class OpenPose(object):
    _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def __init__(self, params):
        self.op = self._libop.newOP(params["logging_level"],
                                    params["output_resolution"],
                                    params["net_resolution"],
                                    params["model_pose"],
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],
                                    params["default_model_folder"])

    def __del__(self):
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display)
        array = np.zeros(shape=(size),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        if display:
            return array, displayImage
        return array

    def poseFromHM(self, image, hm, ratios=[1]):
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")

        # Find largest
        hm_combine = np.zeros(shape=(len(hm), hm[0].shape[1], hm[0].shape[2], hm[0].shape[3]),dtype=np.float32)
        i=0
        for h in hm:
           hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
           i+=1
        hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array, displayImage

    @staticmethod
    def pad_image(image, padValue, bbox):
        h = image.shape[0]
        h = min(bbox[0], h);
        w = image.shape[1]
        bbox[0] = (np.ceil(bbox[0]/8.))*8;
        bbox[1] = max(bbox[1], w);
        bbox[1] = (np.ceil(bbox[1]/8.))*8;
        pad = np.zeros(shape=(4))
        pad[0] = 0;
        pad[1] = 0;
        pad[2] = int(bbox[0]-h);
        pad[3] = int(bbox[1]-w);
        imagePadded = image
        padDown = np.tile(imagePadded[imagePadded.shape[0]-2:imagePadded.shape[0]-1,:,:], [int(pad[2]), 1, 1])*0
        imagePadded = np.vstack((imagePadded,padDown))
        padRight = np.tile(imagePadded[:,imagePadded.shape[1]-2:imagePadded.shape[1]-1,:], [1, int(pad[3]), 1])*0 + padValue
        imagePadded = np.hstack((imagePadded,padRight))
        return imagePadded, pad

    @staticmethod
    def unpad_image(image, padding):
        if padding[2] < 0:
            pass
        elif padding[2] > 0:
            image = image[0:image.shape[0]-int(padding[2]),:]
        if padding[3] < 0:
            pass
        elif padding[3] > 0:
            image = image[:,0:image.shape[1]-int(padding[3])]
        return image

    @staticmethod
    def process_frame(frame, boxsize, padvalue):
        height, width, channels = frame.shape
        scaleImage = float(boxsize) / float(height)
        rframe = cv2.resize(frame, (0,0), fx=scaleImage, fy=scaleImage)
        bbox = [boxsize, max(rframe.shape[1], boxsize)];
        imageForNet, padding = OpenPose.pad_image(rframe, padvalue, bbox)
        imageForNet = imageForNet.astype(float)
        imageForNet = imageForNet/256. - 0.5
        imageForNet = np.transpose(imageForNet, (2,0,1))
        return rframe, imageForNet, padding

if __name__ == "__main__":
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x736"
    params["model_pose"] = "COCO"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 2
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "models/"
    openpose = OpenPose(params)

    img = cv2.imread("examples/media/COCO_val2014_000000000192.jpg")
    arr, output_image = openpose.forward(img, True)
    print arr

    while 1:
        cv2.imshow("output", output_image)
        cv2.waitKey(15)

