import h5py
import numpy as np

base = "mobilenet_"
alphas = ["1_0", "7_5", "5_0", "2_5"]
sizes = [224, 192, 160, 128]
end_str = "_tf.h5"

for alpha in alphas:
    for size in sizes:
        fn = base + alpha + "_" + str(size) + end_str
        print("Working on file : %s" % fn)
        f = h5py.File(fn)
        classification_layer = f.attrs['layer_names'][-3]
        classification_dataset = f[classification_layer]

        weights_name = b'conv_preds/kernel:0'
        bias_name = b'conv_preds/bias:0'

        weights = classification_dataset[weights_name][:]
        bias = classification_dataset[bias_name][:]

        # remove the first class
        weights = weights[..., 1:]
        bias = bias[1:]

        del classification_dataset[weights_name]
        classification_dataset.create_dataset(weights_name, data=weights)

        del classification_dataset[bias_name]
        classification_dataset.create_dataset(bias_name, data=bias)

        f.close()

        print("Finished processing weight file : %s" % (fn))

print("Finished processing all weights")
