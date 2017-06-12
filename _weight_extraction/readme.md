The weight file was obtained from the fantastic repository which contains the tensorflow weights https://github.com/Zehaos/MobileNet.

Since the weight file was in a tensorflow checkpoint, I modified a tensorflow script (which originally only prints the numpy weights, to instead save them onto a file inside the weights folder.

This was then used to manually load the weights into the Keras model.

If you want to run the convertion from checkpoint file to numpy weights, download the checkpoint file and place it in the checkpoint directory.