import numpy as np
import tensorflow as tf
from keras.preprocessing import image as image_utils
import _weight_extraction.mobilenet_v1 as mobilenet_v1
from keras.applications.imagenet_utils import decode_predictions

slim = tf.contrib.slim
checkpoint_file = 'checkpoint/mobilenet_v1_1.0_224.ckpt'


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


with tf.Graph().as_default():
    img_path = '../elephant.jpg'

    img = image_utils.load_img(img_path, target_size=(224, 224))
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        logits, end_points = mobilenet_v1.mobilenet_v1(x, is_training=False, num_classes=1001)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(checkpoint_file,
                                             slim.get_model_variables('MobilenetV1'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([x, probabilities])
        probabilities = probabilities[:, 0:1000]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

    print("End Points : ")
    for k, v in end_points.items():
        print(k, '\t', v)

    print('\nPredicted:', decode_predictions(probabilities))

    graph = sess.graph

    writer = tf.summary.FileWriter('logs/', graph)
    writer.close()
