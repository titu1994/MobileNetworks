# Mobile Networks in Keras

Keras implementation of the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf).

Includes weights trained on ImageNet (obtained from https://github.com/Zehaos/MobileNet, and then converted) in the default model, and a custom layer which performs DepthwiseConvolution2D.

# Benefits of Mobile Nets
As explained in the paper, large neural networks can be exorbitant, both in the amount of memory they require to perform predictions, to the actual size of the model weights.

Therefore, by using Depthwise Convolutions, we can reduce a significant portion of the model size while still retaining very good performance.

# Creating MobileNets

The default MobileNet corresponds to the model pre-trained on ImageNet. It has an input shape of (224, 224, 3).

```
from mobilenets import MobileNets

model = MobileNets()
```

There are two hyperparameters that you can change - `alpha` (the widening factor), and `depth_multiplier`. The ImageNet model uses the default values of 1 for both of the above.

```
from mobilenets import MobileNets

model = MobileNets(alpha=1, depth_multiplier=1)
```

# Testing 

The model can be tested by running the `predict_imagenet.py` script, using the given elephant image. It will return a top 5 prediction score, where "African Elephant" score will be around 96%.

## Conversion of Tensorflow Weights
The weights were originally from https://github.com/Zehaos/MobileNet, which used Tensorflow checkpoints. 

The `_weight_extraction` folder contains instructions on converting a tensorflow checkpoint weights into Numpy arrays, and then load them into Keras weight matrices.
