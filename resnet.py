from keras.layers import Add, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, ZeroPadding2D

class ResNet():
    def _residual_block(self, input, filters, strides=(1, 1), short_cut=False):
        if short_cut:  # convolution block
            short_cut = Conv2D(4 * filters, (1, 1), strides=strides, padding='valid', use_bias=True)(input)
            short_cut = BatchNormalization()(short_cut)
        else:
            short_cut = input  # identity block

        x = Conv2D(filters, (1, 1), strides=strides, padding='valid', use_bias='True')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias='True')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(4 * filters, (1, 1), strides=(1, 1), padding='valid', use_bias='True')(x)
        x = BatchNormalization()(x)

        x = Add()([x, short_cut])
        x = Activation('relu')(x)

        return x

    def _stack(self, input, filters, num_blocks, strides=(2, 2)):
        x = self._residual_block(self, input, filters, strides=strides, short_cut=True)
        for i in range(2, num_blocks):
            x = self._residual_block(x, filters)
        return x

    def resnet(self, inputs, architecture):
        # stage1
        x = ZeroPadding2D((3, 3))(inputs)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=True)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # stage2
        C2 = x = self._stack(x, 64, num_blocks=3, strides=(1, 1))

        # stage3
        C3 = x = self._stack(x, 128, num_blocks=4)

        # stage4
        block_count = {"resnet50": 6, "resnet101": 23}[architecture]
        C4 = x = self._stack(x, 256, num_blocks=block_count)

        # stage5
        C5 = x = self._stack(x, 512, num_blocks=3)

        return x, [C1, C2, C3, C4, C5]
