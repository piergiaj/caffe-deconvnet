import numpy as np
import matplotlib.pyplot as plt
import os

import caffe

caffe_root = '/home/aj/tmp/caffe/'

caffe.set_mode_cpu()
net = caffe.Net('deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

invnet = caffe.Net('invdeploy.prototxt',caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

def norm(x, s=1.0):
    x -= x.min()
    x /= x.max()
    return x*s

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.axis('off')
    plt.imshow(data)


net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('../test/butterfly.jpg'))
out = net.forward()

#feat = net.blobs['conv1'].data[0,:36]
#vis_square(feat, padval=1)
#plt.savefig('conv1_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)

#plt.clf()
#feat = net.blobs['pool5'].data[0]
#vis_square(feat, padval=1)
#plt.savefig('pool5_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)

#plt.clf()
#feat = net.blobs['prob'].data[0]
#plt.plot(feat.flat)
#plt.savefig('prob_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)


for b in invnet.params:
    invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
#    print invnet.params[b][0].data.shape, net.params[b][0].data.shape
#    print invnet.params[b][1].data.shape, net.params[b][1].data.shape
#    invnet.params[b][1].data[...] = net.params[b][1].data.reshape(invnet.params[b][1].data.shape)

feat = net.blobs['pool5'].data
feat[0][feat[0] < 150] = 0
vis_square(feat[0], padval=1)
plt.show()



invnet.blobs['pooled'].data[...] = feat
invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
invnet.forward()


plt.clf()
feat = norm(invnet.blobs['conv1'].data[0],255.0)
plt.imshow(transformer.deprocess('data', feat))
plt.show()

#vis_square(feat, padval=1)
#plt.savefig('test_deconv.png', dpi = 400, bbox_inches='tight', transparent=True)


#features = np.zeros((50000,4096))
#i = 0

# for img in os.listdir('ILSVRC2012_img_val'):
#     net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('ILSVRC2012_img_val/'+img))
#     out = net.forward()
#     feat = net.blobs['fc7'].data[0]
#     features[i,:] = feat
#     i += 1

# np.save('features', features)
