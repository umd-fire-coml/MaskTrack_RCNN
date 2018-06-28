import imageio
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_block, bottleneck

from pwc_net.model import PWCNet


class MaskPropagation(object):

    def __init__(self, mode, config, weights_path, debugging=False):
        self.name = 'maskprop'
        self.mode = mode
        self.config = config
        self.weights_path = weights_path
        self.debugging = debugging

        assert mode in ['training', 'inference']

        self._build()

    def _build(self):

        # set up image and mask inputs
        self.prev_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='prev_image')
        self.curr_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='curr_image')

        def scale_and_expand(v):
            return tf.divide(tf.expand_dims(v, axis=0), 255)

        prev = scale_and_expand(self.prev_image)
        curr = scale_and_expand(self.curr_image)

        # feed images into PWC-Net to get optical flow field
        x, _, _ = PWCNet()(prev, curr)

        self.flow_field = tf.image.resize_bilinear(x, tf.shape(prev)[1:3])

        # TODO freeze optical flow layers to not train

        # feed masks and flow field into CNN (conv5)
        self.prev_masks = tf.placeholder(tf.float32, shape=(None, None, 1), name='prev_masks')
        x = tf.expand_dims(self.prev_masks, axis=0)

        x = tf.concat([x, self.flow_field], axis=3)

        x, _ = self._build_mp_conv5(x, is_training=self.mode == 'training')

        x = tf.layers.conv2d(x, 1, (3, 3), strides=(1, 1), padding='same', activation=tf.sigmoid)

        x = tf.Print(x, [tf.shape(x)])

        # TODO add training losses
        self.propagated_mask = x

        # load weights for optical flow model from disk
        tf.global_variables_initializer()

        sess = K.get_session()
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pwcnet'))

        saver.restore(sess, self.weights_path)

    # The full preactivation 'v2' ResNet variant implemented in this module was
    # introduced by:
    # [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    #     Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
    # NOTE: The pre-activation variant does not have batch
    # normalization or activation functions in the residual unit output. See [2].
    def _build_mp_conv5(self, inputs,
                        is_training=True,
                        output_stride=None,
                        reuse=None):
        """Generator for v2 (preactivation) ResNet 101 model conv5 section.
        Args:
          inputs: A tensor of size [batch, height_in, width_in, channels].
          is_training: whether batch_norm layers are in training mode.
          output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
          reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        Returns:
          net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is 0 or None,
            then net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is a non-zero integer, net contains the
            pre-softmax activations.
          end_points: A dictionary from components of the network to the corresponding
            activation.
        Raises:
          ValueError: If the target output_stride is not valid.
        """

        # blocks: A list of length equal to the number of ResNet blocks. Each element
        # is a resnet_utils.Block object describing the units in the block.
        blocks = [
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]

        scope = 'resnet_v2_101'

        with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, bottleneck,
                                 resnet_utils.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

                    # batch normalization or activation functions (see [2])

                    # Convert end_points_collection into a dictionary of end_points.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)

                    return net, end_points

    def propagate_masks(self, prev_image, curr_image, prev_masks):
        sess = K.get_session()

        inputs = {self.prev_image: prev_image,
                  self.curr_image: curr_image,
                  self.prev_masks: prev_masks}

        mask = sess.run(self.flow_field, feed_dict=inputs)

        return mask

    # TODO add training sections


# test script
def test():
    mp = MaskPropagation('inference', None, '/pwc_net/model_3000epoch/model_3007.ckpt', debugging=True)

    img1 = imageio.imread('../pwc_net/test_images/frame1.jpg')
    img2 = imageio.imread('../pwc_net/test_images/frame2.jpg')

    oflow = mp.propagate_masks(img1, img2, np.reshape(np.empty(img1.shape)[:, :, 0], (1080, 1349, 1)))
    print(oflow.shape)
    plt.figure(1)
    plt.imshow(oflow[0, :, :, 0])
    plt.figure(2)
    plt.imshow(oflow[0, :, :, 1])
    plt.show()
