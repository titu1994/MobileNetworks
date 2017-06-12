import numpy as np
import glob
import os

from mobilenets import MobileNets

weight_path = '_weight_extraction/weights/*.npy'

fns = glob.glob(weight_path)

conv1_bn = fns[0:3]
conv1_wb = fns[3:5]

weight_dict = {}
id = 1

# print(conv1_bn)
# print(conv1_wb)

# for i, fn in enumerate(fns):
#     print(i + 1, fn)

dw9 = fns[5:15]
dw10 = fns[15:25]
dw11 = fns[25:35]
dw12 = fns[35:45]
dw13 = fns[45:55]

dw1 = fns[55:65]
dw2 = fns[65:75]
dw3 = fns[75:85]
dw4 = fns[85:95]
dw5 = fns[95:105]
dw6 = fns[105:115]
dw7 = fns[115:125]
dw8 = fns[125:135]

fcn = fns[135:]

dwlist = [dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, dw13]

for i, dw in enumerate(dwlist):
    print(i + 1, dw)

model = MobileNets((224, 224, 3))

for i, layer in enumerate(model.layers):
    print(i, layer.name)

conv1_weights_biases = [np.load(conv1_wb[1]), np.load(conv1_wb[0])]
model.layers[1].set_weights(conv1_weights_biases)

conv1_bn = [np.load(fn) for fn in conv1_bn]
model.layers[2].set_weights(conv1_bn)

layer_index = 4

for i, dw in enumerate(dwlist):
    dw_weights_biases = [np.load(dw[1]), np.load(dw[0])]
    dw_bn = [np.load(fn) for fn in dw[2:5]]

    pw_weights_biases = [np.load(dw[6]), np.load(dw[5])]
    pw_bn = [np.load(fn) for fn in dw[7:]]

    model.layers[layer_index].set_weights(dw_weights_biases)
    model.layers[layer_index + 1].set_weights(dw_bn)

    model.layers[layer_index + 3].set_weights(pw_weights_biases)
    model.layers[layer_index + 4].set_weights(pw_bn)

    layer_index += 6

fc_weights_bias = [np.load(fcn[1]), np.load(fcn[0])]

model.layers[-1].set_weights(fc_weights_bias)

model.save_weights('mobilenet_imagenet_tf.h5')



