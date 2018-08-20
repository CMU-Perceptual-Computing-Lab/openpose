"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python
"""
import numpy as np
import ctypes as ct
import cv2
import os
from sys import platform
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    os.environ['PATH'] = dir_path + "/../../bin;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Debug;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Release;" + os.environ['PATH']

class OpenPose(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    elif platform == "darwin":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.dylib')
    elif platform == "win32":
        try:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Release/_openpose.dll')
        except OSError as e:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Debug/_openpose.dll')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_bool, ct.c_bool
    ]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.getHandSize.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.int32)
    ]
    _libop.getHandSize.restype = None

    _libop.getHandOutputs.argtypes = [
        ct.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32)
    ]
    _libop.getHandOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
		                            self.encode(params["output_resolution"]),
                                    self.encode(params["net_resolution"]),
                                    self.encode(params["model_pose"]),
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],
                                    self.encode(params["default_model_folder"]),
                                    self.encode(params["hand_net_resolution"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, display=False, hands=False):
        """
        Forward: Takes in an image and returns the human 2D poses, along with drawn image if required

        Parameters
        ----------
        image : color image of type ndarray
        display : If set to true, we return both the pose and an annotated image for visualization
        hands : If set to true, also predict and return hand keypoints.

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        hands : ndarray of hands correspodning to the poses. [2 * People * HandParts * XYConfidence]
        """
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape), dtype=np.uint8)
        size = np.zeros(shape=(3), dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display, hands)

        array = np.zeros(shape=(size), dtype=np.float32)
        self._libop.getOutputs(self.op, array)

        if hands:
            hand_size = np.zeros(shape=(3), dtype=np.int32)
            self._libop.getHandSize(self.op, hand_size)

            hands_array = np.zeros(shape=(2, *hand_size), dtype=np.float32)
            self._libop.getHandOutputs(self.op, hands_array[0], hands_array[1])

        if display and hands:
            return array, displayImage, hands_array
        elif display:
            return array, displayImage
        elif hands:
            return array, hands_array

        return array

    def poseFromHM(self, image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose

        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
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
    def process_frames(frame, boxsize = 368, scales = [1]):
        base_net_res = None
        imagesForNet = []
        imagesOrig = []
        for idx, scale in enumerate(scales):
            # Calculate net resolution (width, height)
            if idx == 0:
                net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
                base_net_res = net_res
            else:
                net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale)+0.5)/16*16))),
                          (min(base_net_res[1], max(1, int((base_net_res[1] * scale)+0.5)/16*16))))
            input_res = [frame.shape[1], frame.shape[0]]
            scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
            warp_matrix = np.array([[scale_factor,0,0],
                                    [0,scale_factor,0]])
            if scale_factor != 1:
                imageForNet = cv2.warpAffine(frame, warp_matrix, net_res, flags=(cv2.INTER_AREA if scale_factor < 1. else cv2.INTER_CUBIC), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                imageForNet = frame.copy()

            imageOrig = imageForNet.copy()
            imageForNet = imageForNet.astype(float)
            imageForNet = imageForNet/256. - 0.5
            imageForNet = np.transpose(imageForNet, (2,0,1))

            imagesForNet.append(imageForNet)
            imagesOrig.append(imageOrig)

        return imagesForNet, imagesOrig

    @staticmethod
    def draw_all(imageForNet, heatmaps, currIndex, div=4., norm=False):
        netDecreaseFactor = float(imageForNet.shape[0]) / float(heatmaps.shape[2]) # 8
        resized_heatmaps = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[1], imageForNet.shape[0], imageForNet.shape[1]))
        num_maps = heatmaps.shape[1]
        combined = None
        for i in range(0, num_maps):
            heatmap = heatmaps[0,i,:,:]
            resizedHeatmap = cv2.resize(heatmap, (0,0), fx=netDecreaseFactor, fy=netDecreaseFactor)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resizedHeatmap)

            if i==currIndex and currIndex >=0:
                resizedHeatmap = np.abs(resizedHeatmap)
                resizedHeatmap = (resizedHeatmap*255.).astype(dtype='uint8')
                im_color = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)
                resizedHeatmap = cv2.addWeighted(imageForNet, 1, im_color, 0.3, 0)
                cv2.circle(resizedHeatmap, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
                return resizedHeatmap
            else:
                resizedHeatmap = np.abs(resizedHeatmap)
                if combined is None:
                    combined = np.copy(resizedHeatmap);
                else:
                    if i <= num_maps-2:
                        combined += resizedHeatmap;
                        if norm:
                            combined = np.maximum(0, np.minimum(1, combined));

        if currIndex < 0:
            combined /= div
            combined = (combined*255.).astype(dtype='uint8')
            im_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(imageForNet, 0.5, im_color, 0.5, 0)
            cv2.circle(combined, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
            return combined


if __name__ == "__main__":
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "../../../models/"
    params["hand_net_resolution"] = "368x368"
    openpose = OpenPose(params)

    img = cv2.imread("../../../examples/media/COCO_val2014_000000000192.jpg")
    arr, output_image = openpose.forward(img, True)
    print(arr)

    while 1:
        cv2.imshow("output", output_image)
        cv2.waitKey(15)
