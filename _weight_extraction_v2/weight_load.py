import numpy as np
import glob
import os

from mobilenets import MobileNetV2

IMG_SIZE = 224
ALPHA = 1.0
model = MobileNetV2((IMG_SIZE, IMG_SIZE, 3), alpha=ALPHA, expansion_factor=6, weights=None)

weight_path = 'weights/*.npy'

fns = sorted(glob.glob(weight_path))

for i, fn in enumerate(fns):
    print("File : ", i, ": ", fn)

conv_final_bn = fns[0:4]  # C1 (bn) : beta, gamma, movine mean, moving var
conv_final_wb = fns[4:5]

conv_zero_bn = fns[5:9]
conv_zero_wb = fns[9:10]
conv_expanded_dw = fns[252:262]

fcn = fns[10:12]

dw10 = fns[12:27]
dw11 = fns[27:42]
dw12 = fns[42:57]
dw13 = fns[57:72]
dw14 = fns[72:87]
dw15 = fns[87:102]
dw16 = fns[102:117]

dw1 = fns[117:132]
dw2 = fns[132:147]
dw3 = fns[147:162]
dw4 = fns[162:177]
dw5 = fns[177:192]
dw6 = fns[192:207]
dw7 = fns[207:222]
dw8 = fns[222:237]
dw9 = fns[237:252]

print("\nInitial Conv-BN : ", conv_final_wb, conv_final_bn)
print("Conv : ", conv_zero_wb, conv_zero_bn)
print("Expand Conv : ", conv_expanded_dw)

dwlist = [dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, dw13, dw14, dw15, dw16]

for i, dw in enumerate(dwlist):
    print('DW block %d' % (i + 1), dw)

print("Final dense layer", fcn)
print()

for i, layer in enumerate(model.layers):
    print(i, layer.name)

# used for sanity check that all weights have been loaded
layers = model.layers
layer_has_weights = [(True if len(layer.get_weights()) != 0 else False)
                     for layer in model.layers]
layer_weights_saved = [False for _ in range(len(model.layers))]

print()
num_weights = ([(len(layer.get_weights()), layer.name) for layer in model.layers])
counter = 0
for count, name in num_weights:
    if count != 0:
        # print(name, count)
        counter += count

print("Number of weight matrices : ", counter)

# these two layers will be loaded in the next line
layer_weights_saved[1] = True
layer_weights_saved[2] = True

conv1_weights = [np.load(conv_zero_wb[0])]
model.layers[1].set_weights(conv1_weights)

conv1_bn = [np.load(fn) for fn in conv_zero_bn]
model.layers[2].set_weights([conv1_bn[1], conv1_bn[0], conv1_bn[2], conv1_bn[3]])
print("\nLoaded initail conv weights")

layer_index = 4

dw_weights = [np.load(conv_expanded_dw[4])]
dw_bn = [np.load(fn) for fn in conv_expanded_dw[0:4]]

pw_weights_biases = [np.load(conv_expanded_dw[-1])]
pw_bn = [np.load(fn) for fn in conv_expanded_dw[5:-1]]

model.layers[layer_index].set_weights(dw_weights)
model.layers[layer_index + 1].set_weights([dw_bn[1], dw_bn[0], dw_bn[2], dw_bn[3]])

model.layers[layer_index + 3].set_weights(pw_weights_biases)
model.layers[layer_index + 4].set_weights([pw_bn[1], pw_bn[0], pw_bn[2], pw_bn[3]])

layer_weights_saved[4] = True
layer_weights_saved[5] = True
layer_weights_saved[7] = True
layer_weights_saved[8] = True


# now loop through the internal expand-dw-pw layers
layer_index = 9

for i, dw in enumerate(dwlist):
    dw_weights = [np.load(dw[4])]
    dw_bn = [np.load(fn) for fn in dw[0:4]]

    expand_weights = [np.load(dw[9])]
    expand_bn = [np.load(fn) for fn in dw[5:9]]

    pw_weights_biases = [np.load(dw[-1])]
    pw_bn = [np.load(fn) for fn in dw[10:-1]]

    model.layers[layer_index].set_weights(expand_weights)
    model.layers[layer_index + 1].set_weights([expand_bn[1], expand_bn[0], expand_bn[2], expand_bn[3]])

    model.layers[layer_index + 3].set_weights(dw_weights)
    model.layers[layer_index + 4].set_weights([dw_bn[1], dw_bn[0], dw_bn[2], dw_bn[3]])

    model.layers[layer_index + 6].set_weights(pw_weights_biases)
    model.layers[layer_index + 7].set_weights([pw_bn[1], pw_bn[0], pw_bn[2], pw_bn[3]])

    # for sanity check, set True for all layers whise weights were changed
    layer_weights_saved[layer_index] = True
    layer_weights_saved[layer_index + 1] = True
    layer_weights_saved[layer_index + 3] = True
    layer_weights_saved[layer_index + 4] = True
    layer_weights_saved[layer_index + 6] = True
    layer_weights_saved[layer_index + 7] = True

    print('Loaded Expand-DW-PW layer %d weights' % (i + 1))
    layer_index += 8

    if 'add' in model.layers[layer_index].name:
        layer_index += 1

final_conv_bn = [np.load(fn) for fn in conv_final_bn]
final_conv_wb = [np.load(conv_final_wb[0])]

model.layers[layer_index].set_weights(final_conv_wb)
model.layers[layer_index + 1].set_weights([final_conv_bn[1], final_conv_bn[0], final_conv_bn[2], final_conv_bn[3]])

layer_weights_saved[layer_index] = True
layer_weights_saved[layer_index + 1] = True

layer_index += 3

fc_weights_bias = [np.load(fcn[1]), np.load(fcn[0])]
fc_weights_bias[0] = fc_weights_bias[0][..., 1:]
fc_weights_bias[1] = fc_weights_bias[1][1:]
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
