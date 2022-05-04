from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Add, UpSampling2D
from keras import backend


class Hourglass():
    num_stacks = 4
    
    # bottom is input, _skip is short_cut
    def _bottleneck_block(self, input, num_out_channels, block_name):
        if backend.int_shape(input)[-1] == num_out_channels:
            short_cut = input
        else:
            short_cut = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                        name=block_name + 'skip')(input)

        # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
        x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(input)
        x = BatchNormalization()(x)
        x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(x)
        x = BatchNormalization()(x)

        x = Add(name=block_name + '_residual')([short_cut, x])

        return x

    def _front_module(self, input, num_channels):
        # front module, input to 1/4 resolution
        # one 7*7 conv and maxpooling, three residual block

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same',
                name='front_conv_7x7')(input)
        x = BatchNormalization()(x)

        x = self._bottleneck_block(x, num_channels // 2, 'front_residual_x1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = self._bottleneck_block(x, num_channels // 2, 'front_residual_x2')
        x = self._bottleneck_block(x, num_channels, 'front_residual_x3')

        return x

    def _hourglass_leftside(self, input, hglayer, num_channels):
        # create left half blocks for hourglass module
        # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

        hgname = 'hg' + str(hglayer)

        f1 = self._bottleneck_block(input, num_channels, hgname + '_11')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

        f2 = self._bottleneck_block(x, num_channels, hgname + '_12')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

        f4 = self._bottleneck_block(x, num_channels, hgname + '_14')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

        f8 = self._bottleneck_block(x, num_channels, hgname + '_18')

        return (f1, f2, f4, f8)

    def _connect_left_n_right(self, left, right, name, num_channels):
        # left -> 1 bottlenect
        # right -> upsampling
        # Add   -> left + right   

        x_left = self._bottleneck_block(left, num_channels, name + '_connect')
        x_right = UpSampling2D()(right)
        add = Add()([x_left, x_right])
        output = self._bottleneck_block(add, num_channels, name + '_connect_conv')

        return output

    def _bottom_layer(self, l_f8, hgid, num_channels):
        l_f8_connect = self._bottleneck_block(l_f8, num_channels, str(hgid) + '_lf8')

        x = self._bottleneck_block(l_f8, num_channels, str(hgid) + '_lf8_x1')
        x = self._bottleneck_block(x, num_channels, str(hgid) + '_lf8_x2')
        x = self._bottleneck_block(x, num_channels, str(hgid) + '_lf8_x3')

        r_f8 = Add()([x, l_f8_connect])

        return r_f8

    def _hourglass_rightside(self, left_features, hglayer, num_channels):
        l_f1, l_f2, l_f4, l_f8 = left_features

        r_f8 = self._bottom_layer(l_f8, hglayer, num_channels)

        r_f4 = self._connect_left_n_right(l_f4, r_f8, 'hg' + str(hglayer) + '_rf4', num_channels)

        r_f2 = self._connect_left_n_right(l_f2, r_f4, 'hg' + str(hglayer) + '_rf2', num_channels)

        r_f1 = self._connect_left_n_right(l_f1, r_f2, 'hg' + str(hglayer) + '_rf1', num_channels)

        return r_f1

    def _create_heads(self, prelayer_features, r_f1, num_classes, hgid, num_channels):
        # two heads, one head to next stage, one head to intermediate features.
        head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same',
                    name=str(hgid) + '_conv_1x1_x1')(r_f1)
        head = BatchNormalization()(head)

        # for head as intermediate supervision, use 'linear' as an activation function.
        head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                            name=str(hgid) + '_conv_1x1_parts')(head)
        
        # use linear activation
        head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hgid) + '_conv_1x1_2')(head)
        head_m = Conv2D(num_channels, kernel_size = (1, 1), activation='linear', padding='same',
                        name=str(hgid) + '_conv_1x1_x3')(head_parts)
        
        head_next_stage = Add()([head, head_m, prelayer_features])
        
        return head_next_stage

    def _hourglass_module(self, input, num_classes, num_channels, hgid):
        # build left side of hourglass module and [f1, f2, f4, f8]
        left_features = self._hourglass_leftside(input, hgid, num_channels)

        # create right side of hourglass module and connect with features from left side
        r_f1 = self._hourglass_rightside(left_features, hgid, num_channels)

        # add 1*1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        head_next_stage = self._create_heads(input, r_f1, num_classes, hgid, num_channels)

        return head_next_stage

    def hourglass_model(self, inputs, num_classes, num_channels):
        front_features = self._front_module(inputs, num_channels)
        head_next_stage = front_features
        for i in range(self.num_stacks):
            head_next_stage = self._hourglass_module(head_next_stage, num_classes, num_channels, i)
        
        return head_next_stage