from keras import backend as K
from keras import layers as KL
from keras import data as KD

class TrajectoryModule(object):

    name = 'trajectory'
    relu_max = 6

    def __init__(self, mode, config, model_dir,
        debugging=False, optimizer=keras.optimizers.Adadelta(),
        loss_function=keras.losses.binary_cross_entropy):
        """
        Creates and builds the mask propagation network.
        :param mode: either 'training' or 'inference'
        :param config: not used atm
        :param pwc_net_weights_path: path to the weights for pwc-net
        :param model_dir: directory to save/load logs and model checkpoints
        :param debugging: whether to include extra print operations
        :param isolated: whether this is the only network running
        :param optimizer: tf optimizer object,
          e.g. tf.train.AdadeltaOptimizer(), tf.train.AdamOptimizer()
        :param loss_function: tf function that computes the loss between
          the predicted and gt masks
        """
        self.mode = mode
        self.model_dir = model_dir
        self.config = config
        self.weights_path = config.weights_path if config else None

        self.debugging = debugging

        self.optimizer = optimizer
        self.loss_function = loss_function

        assert mode in ['training', 'inference']

        self.keras_model = self._build()

    def _build(self):
        """
        Builds the computation graph for the mask propagation network.
        """

        flow_field = Input(shape=(None, None, None, 2))
        prev_mask = Input(shape=(None, None, None, 1))
        inputs = [flow_field, prev_mask]

        x = KL.Concatenate(inputs, axis=3)

        # build the u-net and get the final propagated mask
        x = self._build_unet(x)
        self.propagated_masks = x

        model = KM.Model(inputs, outputs, name='mask_rcnn')

        model = KM.Model(inputs, outputs, name='mask_rcnn')

        if self.weights_path:

            model.load_weights(weights_path)

        return model

    def m_relu(x):

        return K.relu(x, max_value=relu_max)

    def _build_unet(self, x, conv_act=m_relu, deconv_act=None):
        """
        Builds the mask propagation network proper
          (based on the u-Net architecture).
        :param x: input tensor of the mask and flow field concatenated
          [batch, w, h, 1+2]
        :param conv_act: activation function for the convolution layers
        :param deconv_act: activation function for
          the transposed convolution layers
        :return: output tensor of the U-Net [batch, w, h, 1]
          As a side effect, two instance variables unet_left_wing and
          unet_right_wing are set with the final output tensors
          for each layer of the two halves of the U.
        """

        _input = x

        x = KL.Conv2D(64, (3, 3), activation=conv_act, name='L1_conv1')(x)
        x = KL.Conv2D(64, (3, 3), activation=conv_act, name='L1_conv2')(x)
        L1 = x

        x = KL.MaxPooling2D( (2, 2), (2, 2), name='L2_pool')(x)
        x = KL.Conv2D( 128, (3, 3), activation=conv_act, name='L2_conv1')(x)
        x = KL.Conv2D( 128, (3, 3), activation=conv_act, name='L2_conv2')(x)
        L2 = x

        x = KL.MaxPooling2D( (2, 2), (2, 2), name='L3_pool')
        x = KL.Conv2D( 256, (3, 3), activation=conv_act, name='L3_conv1')(x)
        x = KL.Conv2D( 256, (3, 3), activation=conv_act, name='L3_conv2')(x)
        L3 = x

        x = KL.MaxPooling2D( (2, 2), (2, 2), name='L4_pool')
        x = KL.Conv2D( 512, (3, 3), activation=conv_act, name='L4_conv1')(x)
        x = KL.Conv2D( 512, (3, 3), activation=conv_act, name='L4_conv2')(x)
        L4 = x

        x = KL.MaxPooling2D( (2, 2), (2, 2), name='L5_pool')(x)
        x = KL.Conv2D( 1024, (3, 3), activation=conv_act, name='L5_conv1')(x)
        x = KL.Conv2D( 1024, (3, 3), activation=conv_act, name='L5_conv2')(x)
        L5 = x

        x = KL.Conv2DTranspose( 1024, (2, 2), strides=(2, 2),
            activation=deconv_act, name='P4_upconv')(x)
        x = tf.concat([L4, tf.image.resize_images( tf.shape(L4)[1:3])],
                      axis=3, name='P4_concat')(x)
        x = KL.Conv2D( 512, (3, 3), activation=conv_act, name='P4_conv1')(x)
        x = KL.Conv2D( 512, (3, 3), activation=conv_act, name='P4_conv2')(x)
        P4 = x

        x = KL.Conv2DTranspose( 512, (2, 2), strides=(2, 2),
            activation=deconv_act, name='P3_upconv')(x)
        x = tf.concat([L3, tf.image.resize_images( tf.shape(L3)[1:3])],
                      axis=3, name='P3_concat')(x)
        x = KL.Conv2D( 256, (3, 3), activation=conv_act, name='P3_conv1')(x)
        x = KL.Conv2D( 256, (3, 3), activation=conv_act, name='P3_conv2')(x)
        P3 = x

        x = KL.Conv2DTranspose( 256, (2, 2), strides=(2, 2),
            activation=deconv_act, name='P2_upconv')(x)
        x = tf.concat([L2, tf.image.resize_images( tf.shape(L2)[1:3])],
                      axis=3, name='P2_concat')(x)
        x = KL.Conv2D( 128, (3, 3), activation=conv_act, name='P2_conv1')(x)
        x = KL.Conv2D( 128, (3, 3), activation=conv_act, name='P2_conv2')(x)
        P2 = x

        x = KL.Conv2DTranspose( 128, (2, 2), strides=(2, 2),
            activation=deconv_act, name='P1_upconv')(x)
        x = tf.concat([L1, tf.image.resize_images( tf.shape(L1)[1:3])],
                      axis=3, name='P1_concat')(x)
        x = KL.Conv2D( 64, (3, 3), activation=conv_act, name='P1_conv1')(x)
        x = KL.Conv2D( 64, (3, 3), activation=conv_act, name='P1_conv2')(x)
        P1 = x

        x = tf.image.resize_images( tf.shape(_input)[1:3])(x)
        x = KL.Conv2D( 1, (1, 1), activation=tf.sigmoid, name='P0_conv')(x)

        self.unet_left_wing = [L1, L2, L3, L4, L5]
        self.unet_right_wing = [P4, P3, P2, P1]

        return x

    def compile(self):

        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss_function)

    def train_batch(self, prev_images, curr_images, prev_masks, gt_masks):
        """
        Trains the mask propagation network on a single batch of inputs.
        :param prev_images: previous images at time t-1 [batch, w, h, 3]
        :param curr_images: current images at time t [batch, w, h, 3]
        :param prev_masks: the masks at time t [batch, w, h, 1]
        :param gt_masks: ground truth next masks at time t+1 [batch, w, h, 1]
        :return: batch loss of the predicted masks against the provided ground truths
        """
        assert self.mode == 'training'

        inputs = {self.prev_images: prev_images,
                  self.curr_images: curr_images,
                  self.prev_masks: prev_masks,
                  self.gt_masks: gt_masks}

        _, loss = self.sess.run([self.optimizable, self.loss], feed_dict=inputs)

        return loss

    #MAJOR WORK IN PROGRESS
    def train_multi_step(self, generator, steps, batch_size, output_types,
        output_shapes=None):
        """DO NOT USE. THIS IS VERY BAD IF YOU USE. DO NOT USE
        Trains the mask propagation network on multiple steps (batches).
        (Essentially an epoch.)
        :param train_dataset: Training Dataset object
        :param steps: Number of times to call the generator.
          (Number of steps in this "epoch".)
        :param batch_size: A tf.int64 scalar tf.Tensor, representing the number
          of consecutive elements of the generator to combine in a single batch.
        :param output_types: output_types: A nested structure of
          tf.DType objects corresponding to each component of an element yielded
          by generator.
        :param output_shapes: (Optional.) A nested structure of tf.TensorShape
          objects corresponding to each component of an element yielded by
          generator.
        :return: a list of batch losses of the predicted masks against the
          generated ground truths
        """
        assert self.mode == 'training'
        assert isinstance(generator, TensorflowDataGenerator)

        dataset = TD.Dataset().batch(batch_size).from_generator(generator,
                                                   output_types=output_types, 
                                                   output_shapes=output_shapes)
        _iter = dataset.make_initializable_iterator()
        element = _iter.get_next()
        self.sess.run(_iter.initializer)

        sliced_tensor = generator.slice_tensor(element)
        inputs = {self.prev_image: sliced_tensor['prev_image'],
                  self.curr_image: sliced_tensor['curr_image'],
                  self.prev_mask: sliced_tensor['prev_mask'],
                  self.gt_mask: sliced_tensor['gt_mask']}

        losses = [None] * steps

        for i in range(steps):
             _, loss = self.sess.run([self.optimizer, self.loss],
                                     feed_dict=inputs)
             losses.append(loss)

        return losses

    def inference(self, prev_image, curr_image):
        """
        Evaluates the model to get the flow field between the two images.
        :param prev_image: starting image for flow [w, h, 3]
        :param curr_image: ending image for flow [w, h, 3]
        :return: flow field for the images [batch, w, h, 2]
        """

        assert self.mode == 'inference'

        inputs = {self.prev_images: np.expand_dims(prev_image, 0),
                  self.curr_images: np.expand_dims(curr_image, 0)}

        mask = self.sess.run(self.flow_field, feed_dict=inputs)

        return mask

    def save_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement saving all weights
        pass

    def load_weights(self, filename):
        weights_pathname = os.path.join(self.model_dir, filename)

        # TODO implement loading all weights
        pass
