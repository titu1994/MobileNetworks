import h5py
from shutil import copyfile

'''
Change fn path here and then execute script to create a weight 
file with the last 5 layers removed.
'''

base = "mobilenet_v2_"
alphas = ["1_0", "3_5", "5_0", "7_5"]
sizes = [96, 128, 160, 192, 224]
end_str = "_tf.h5"

# last 2 largest models
# alphas = ["1_3", "1_4"]  # alphas = ["1_3", "1_4"]
# sizes = [224]  # sizes = [224]

for alpha in alphas:
    for size in sizes:
        fn = base + alpha + "_" + str(size) + end_str
        print("Working on file : %s" % fn)

        new_fn = fn[:-3] + "_no_top.h5"
        copyfile(fn, new_fn)

        f = h5py.File(new_fn)

        layer_names_keep = f.attrs['layer_names'][:-6]
        layer_names_drop = f.attrs['layer_names'][-6:]

        for fn in layer_names_drop:
            del f[fn]

        f.attrs['layer_names'] = layer_names_keep
        f.close()
        print("Created 'No-Top' Weights for %s" % fn)




