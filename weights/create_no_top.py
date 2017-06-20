import h5py
from shutil import copyfile

'''
Change fn path here and then execute script to create a weight 
file with the last 5 layers removed.
'''

base = "mobilenet_"
alphas = ["1_0",]
sizes = [160]
end_str = "_tf.h5"

for alpha in alphas:
    for size in sizes:
        fn = base + alpha + "_" + str(size) + end_str
        print("Working on file : %s" % fn)

        new_fn = fn[:-3] + "_no_top.h5"
        copyfile(fn, new_fn)

        f = h5py.File(new_fn)

        layer_names_keep = f.attrs['layer_names'][:-5]
        layer_names_drop = f.attrs['layer_names'][-5:]

        for fn in layer_names_drop:
            del f[fn]

        f.attrs['layer_names'] = layer_names_keep
        f.close()
        print("Created 'No-Top' Weights for %s" % fn)




