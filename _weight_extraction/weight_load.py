import numpy as np
import glob
import os

from mobilenets import MobileNets

model = MobileNets((128, 128, 3), alpha=0.25, weights=None)

weight_path = 'weights/*.npy'

fns = glob.glob(weight_path)

for i, fn in enumerate(fns):
    print("File : ", i + 1, ": ", fn)

conv1_bn = fns[0:4]
conv1_wb = fns[4:5]

dw10 = fns[5:15]
dw11 = fns[15:25]
dw12 = fns[25:35]
dw13 = fns[35:45]

dw1 = fns[45:55]
dw2 = fns[55:65]
dw3 = fns[65:75]
dw4 = fns[75:85]
dw5 = fns[85:95]
dw6 = fns[95:105]
dw7 = fns[105:115]
dw8 = fns[115:125]
dw9 = fns[125:135]

fcn = fns[135:]

print("\nInitial Conv-BN : ", conv1_wb, conv1_bn)

dwlist = [dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, dw13]

for i, dw in enumerate(dwlist):
    print('DW block %d' % (i + 1), dw)

print("Final dense layer", fcn)
print()

#for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# used for sanity check that all weights have been loaded
layers = model.layers
layer_has_weights = [(True if len(layer.get_weights()) != 0 else False)
                     for layer in model.layers]
layer_weights_saved = [False for _ in range(len(model.layers))]

# these two layers will be loaded in the next line
layer_weights_saved[1] = True
layer_weights_saved[2] = True

conv1_weights = [np.load(conv1_wb[0])]
model.layers[1].set_weights(conv1_weights)

conv1_bn = [np.load(fn) for fn in conv1_bn]
model.layers[2].set_weights([conv1_bn[1], conv1_bn[0], conv1_bn[2], conv1_bn[3]])
print("\nLoaded initail conv weights")

layer_index = 4

for i, dw in enumerate(dwlist):
    dw_weights = [np.load(dw[4])]
    dw_bn = [np.load(fn) for fn in dw[0:4]]

    pw_weights_biases = [np.load(dw[-1])]
    pw_bn = [np.load(fn) for fn in dw[5:-1]]

    model.layers[layer_index].set_weights(dw_weights)
    model.layers[layer_index + 1].set_weights([dw_bn[1], dw_bn[0], dw_bn[2], dw_bn[3]])

    model.layers[layer_index + 3].set_weights(pw_weights_biases)
    model.layers[layer_index + 4].set_weights([pw_bn[1], pw_bn[0], pw_bn[2], pw_bn[3]])

    # for sanity check, set True for all layers whise weights were changed
    layer_weights_saved[layer_index] = True
    layer_weights_saved[layer_index + 1] = True
    layer_weights_saved[layer_index + 3] = True
    layer_weights_saved[layer_index + 4] = True

    print('Loaded DW layer %d weights' % (i + 1))
    layer_index += 6

fc_weights_bias = [np.load(fcn[1]), np.load(fcn[0])]
model.layers[-3].set_weights(fc_weights_bias)
print("Loaded final conv classifier weights")

layer_weights_saved[-3] = True

model.save_weights('../weights/mobilenet_imagenet_tf.h5', overwrite=True)
print("Model saved")

# perform check that all weights that could have been loaded, Have been loaded!
print('\nBegin sanity check...')
for layer_id, has_weights in enumerate(layer_has_weights):
    if has_weights and not layer_weights_saved[layer_id]:
        # weights were not saved! report
        print("Layer id %d (%s) weights were not saved!" % (layer_id, model.layers[layer_id].name))

print("Sanity check complete!")

for fn in glob.glob(weight_path):
    os.remove(fn)
