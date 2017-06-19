# Mobile Networks in Keras

Keras implementation of the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf).

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

<table>
<tr align='center'>
<td>Image</td>
<td>Predictions</td>
</tr>
<tr align='left'>
<td>
<img src="https://github.com/titu1994/MobileNetworks/blob/master/elephant.jpg?raw=true" width=100% height=100%>
</td>
<td>
('Indian_elephant', 0.97889256), <br>
('African_elephant', 0.014472792), <br>
('tusker', 0.0066345367), <br>
('warthog', 9.1309346e-08), <br>
('ram', 1.1704416e-08)
</td>
</tr>
</table>

## Conversion of Tensorflow Weights
The weights were originally from https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md, which used Tensorflow checkpoints. 

There are scripts and some documentation for how the weights were converted in the `_weight_extraction` folder.
