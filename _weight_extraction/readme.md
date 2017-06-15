The weight file was obtained from the fantastic repository which contains the tensorflow weights https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

Since the weight file was in a tensorflow checkpoint, I modified a tensorflow script (which originally only prints the numpy weights, to instead save them onto a file inside the weights folder).

This was then used to manually load the weights into the Keras model.

If you want to run the convertion from checkpoint file to numpy weights,
- download the checkpoint file (MobileNet_v1_1.0_224 only for now)
- extract the 3 files in the checkpoint directory
- delete the `global_step` file
- delete all files with `RMSPROP` in their name
- delete all files with `EXPONENTIAL_MOVING_AVERAGE` in their name
- run extract_weights.py
- run weights_load.py