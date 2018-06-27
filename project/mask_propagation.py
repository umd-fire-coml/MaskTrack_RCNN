import keras.layers as KL
import keras.models as KM
import tensorflow as tf
from tensorflow.contrib.slim.nets.resnet_v2 import resnet_v2, resnet_v2_block

from mrcnn.model import conv_block, identity_block
from pwc_net.model import PWCNet

def _conv5(inputs,
           num_classes=None,
           is_training=True,
           global_pool=True,
           output_stride=None,
           reuse=tf.AUTO_REUSE ,
           scope='resnet_v2_101'):
  blocks = [
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=False,
                   reuse=reuse, scope=scope)

class MaskPropagation(object):

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        assert mode in ['training', 'inference']

        self.tf_model = self._build()

    def _build(self):
        # set up image and mask inputs
        prev_image = KL.Input(shape=(None, None, 3))
        curr_image = KL.Input(shape=(None, None, 3))

        scaled_down = KL.Lambda(lambda x: tf.divide(x, 255))

        prev = scaled_down(prev_image)
        curr = scaled_down(curr_image)

        expand = KL.Lambda(lambda x: tf.expand_dims(x[0], axis=0))

        prev = expand(prev)
        curr = expand(curr)

        #def add_history(x):
        #    inbound_layer, _, _ = x._keras_history
        #    inbound_layer.outbound_nodes = []
        #    return x

        #inbound_layer, _, _ = prev._keras_history
        #inbound_layer.outbound_nodes = []

        #prev = tf.map_fn(add_history, tf.convert_to_tensor([prev, curr]))

        # feed images into PWC-Net to get optical flow field
        flow_field, _, _ = PWCNet()(prev, curr)
        print(flow_field)
        # flow_field = flow_field[0]
        #print(flow_field)

        # feed masks and flow field into CNN (conv5)
        
        x, not_sure_what_this_is = _conv5(x)
        
#         prev_masks = KL.Input(batch_shape=(1, None, None, 1))

#         x = KL.concatenate([prev_masks, flow_field], axis=3)
#         x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
#         x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#         mask_prop_conv = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        # return model
        # mp_model = KM.Model(inputs=[prev_image, curr_image, prev_masks],
                            #outputs=[mask_prop_conv])
        mp_model = x

        return mp_model


# test script
mp = MaskPropagation('training', None)
