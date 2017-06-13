from __future__ import print_function
from __future__ import absolute_import
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

import numpy as np
from mobilenets import MobileNets


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNets()

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    # decode predictions does not like the 1001th class (UNKNOWN class),
    # thats why we remove the last prediction and feed it to decode predictions
    preds = preds[:, 0:1000]

    print('Predicted:', decode_predictions(preds))

