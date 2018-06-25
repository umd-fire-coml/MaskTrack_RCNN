import keras.layers as KL
import keras.models as KM


class MaskPropagation(object):
    
    class BatchNorm(KL.BatchNormalization):
        """Batch Normalization class. Subclasses the Keras BN class and
        hardcodes training=False so the BN layer doesn't update
        during training.

        Batch normalization has a negative effect on training if batches are small
        so we disable it here.
        """

        def call(self, inputs, training=None):
            return super(self.__class__, self).call(inputs, training=False)
    
    def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x

    def build(self, mode):
        assert mode in ['training', 'inference']

        # set up image and mask inputs
        curr_image = KL.Input(shape=(None, None, 3))
        prev_image = KL.Input(shape=(None, None, 3))
        prev_masks = KL.Input(shape=(None, None, 1))

        # feed images through PWC-Net for optical flow

        # feed masks and flow field into CNN
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        # return model
        mp_model = KM.Model(inputs=[], outputs=[])

        return mp_model

