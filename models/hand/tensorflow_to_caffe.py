import caffe
import cPickle as cp
import numpy as np

tf_model = '/media/posefs0c/Users/donglaix/Experiments/freiberg_hand3d/weights/cpm_tf.pickle'
with open(tf_model, 'rb') as f:
    tf_weights = cp.load(f)

caffe.set_mode_cpu()
net = caffe.Net('../../../models/hand/pose_deploy.prototxt', caffe.TEST)

for name, weight in tf_weights.iteritems():
    caffe_name = name.split('/')[1]
    wb = name.split('/')[2]
    assert caffe_name in net.params

    if wb == 'weights':
        net.params[caffe_name][0].data[:] = np.transpose(weight, (3,2,1,0))[:]
    else:
        net.params[caffe_name][1].data[:] = weight[:]

net.params['conv1_1'][0].data[:, :, :, :] = net.params['conv1_1'][0].data[:, ::-1, :, :]
for i in (1, 5, 9, 13, 17):
    net.params['Mconv7_stage6'][0].data[i:i+4, :, :, :] = net.params['Mconv7_stage6'][0].data[i+3:i-1:-1, :, :, :]
    net.params['Mconv7_stage6'][1].data[i:i+4] = net.params['Mconv7_stage6'][1].data[i+3:i-1:-1]

net.save('../../../models/hand/pose_tf.caffemodel')