import h5py
from shutil import copyfile

fn = "mobilenet_2_5_128_tf.h5"
new_fn = fn[:-3] + "_no_top.h5"
copyfile(fn, new_fn)

f = h5py.File(new_fn)

layer_names = f.attrs['layer_names'][-5:]
print(layer_names)

for fn in layer_names:
    del f[fn]

f.close()
print("Created 'No-Top' Weights")
