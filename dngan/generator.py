import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl

from dngan import blocks


def dncnn_generator_model(in_channels=3, out_channels=3, **kwargs):
    inp = kl.Input(shape=(None, None, in_channels))
    oup = dncnn_generator(out_channels, **kwargs)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def dncnn_generator(out_channels=3, num_conv=19, internal_residuals=False,
                    global_residual=True):
    if num_conv % 2 != 0 and internal_residuals:
        raise ValueError("Internal residuals requires an even amount of convolutional layers.")

    def call(x):
        inp = x

        # First Conv layer without BN
        x = kl.Conv2D(64, 3, padding='same', name='conv_1')(x)
        x = kl.ReLU(name='relu_1')(x)

        # Remember output if we want residuals
        if internal_residuals:
            x_res = x

        # Add internal Conv layer blocks
        for idx in range(2, num_conv):
            x = kl.Conv2D(64, 3, padding='same', name=f'conv_{idx}')(x)
            x = kl.BatchNormalization(name=f'bn_{idx}')(x)
            x = kl.ReLU(name=f'relu_{idx}')(x)

            # Make residual
            if internal_residuals and idx % 2 == 1:
                x = x + x_res
                x_res = x

        # Last Conv layer without BN and Activation
        x = kl.Conv2D(out_channels, 3, padding='same', name=f'conv_{num_conv}')(x)

        # Global residual
        if global_residual:
            x = x + inp

        # Clip between 0 and 1
        return tf.clip_by_value(x, 0, 1)

    return call


def unet_generator_model(in_channels=3, out_channels=3, **kwargs):
    inp = kl.Input(shape=(None, None, in_channels))
    oup = unet_generator(out_channels, **kwargs)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def unet_generator(out_channels=3, num_levels=3, filter_base=32, conv_per_block=2,
                   internal_residuals=False, global_residual=True):
    # TODO make activation function configurable
    # TODO more internal residuals?

    def filters(level):
        return filter_base * (2**level)

    def call(x):
        inp = x

        # Pad the input to a multiple of num_levels^2
        patch_size = 2**num_levels
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        pad_h = -h % patch_size
        pad_w = -w % patch_size
        padding = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        x = tf.pad(x, padding, mode='CONSTANT')

        # First Conv layer without BN
        x = kl.Conv2D(filter_base, 3, padding='same', name='conv_first')(x)
        x = kl.ReLU(name='relu_1')(x)

        x_level = []
        # Downsampling blocks
        for level in range(num_levels-1):
            # Conv layers for this level
            for idx in range(conv_per_block):
                conv = blocks.conv2d_block(filters(level), kernel_size=3, padding='same',
                                           activation='relu', batch_normalization=True,
                                           name=f'down_{level}/b_{idx}')

                if internal_residuals and idx > 0:
                    x = blocks.residual_block([conv])(x)
                else:
                    x = conv(x)

            # Skip connection
            x_level.append(x)

            # Downsampling
            x = kl.MaxPooling2D(2, name=f'down_{level}/max_pool')(x)

        # Center
        for idx in range(conv_per_block):
            x = blocks.conv2d_block(filters(num_levels-1), kernel_size=3, padding='same',
                                    activation='relu', batch_normalization=True,
                                    name=f'center/b_{idx}')(x)

        # Upsampling blocks
        for level in range(num_levels-2, -1, -1):
            # Upsampling
            # TODO check if this is correct
            x = kl.Conv2DTranspose(filters(level), kernel_size=3, strides=2, padding='same',
                                   name=f'up_{level}/conv_transpose')(x)

            # Skip connection
            x = tf.concat([x, x_level[level]], axis=-1,
                          name=f'up_{level}/skip')

            # Conv layers for this level
            for idx in range(conv_per_block):
                conv = blocks.conv2d_block(filters(level), kernel_size=3, padding='same',
                                           activation='relu', batch_normalization=True,
                                           name=f'up_{level}/b_{idx}')

                if internal_residuals and idx > 0:
                    x = blocks.residual_block([conv])(x)
                else:
                    x = conv(x)

        # Last Conv layer without BN and Activation
        x = kl.Conv2D(out_channels, 3, padding='same', name='conv_last')(x)

        # Crop back to the original size
        x = x[:, :h, :w, :]

        # Global residual
        if global_residual:
            x = x + inp[..., :out_channels]

        # Clip between 0 and 1
        return tf.clip_by_value(x, 0, 1)

    return call


def drunet_generator_model(in_channels=3, out_channels=3, **kwargs):
    inp = kl.Input(shape=(None, None, in_channels))
    oup = drunet_generator(out_channels, **kwargs)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def drunet_generator(out_channels=3, filters=None, num_blocks=4, final_clip=True):

    if filters is None:
        filters = [64, 128, 256, 512]

    use_bias = False
    num_levels = len(filters)

    def pad_input(x):
        # Pad the input to a multiple of num_levels^2
        patch_size = 2**num_levels
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        pad_h = -h % patch_size
        pad_w = -w % patch_size
        padding = [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        return tf.pad(x, padding, mode='CONSTANT'), (h, w)

    def res_block(x, level):
        x_in = x
        x = kl.Conv2D(filters[level], kernel_size=3, use_bias=use_bias, padding='SAME')(x)
        x = kl.ReLU()(x)
        x = kl.Conv2D(filters[level], kernel_size=3, use_bias=use_bias, padding='SAME')(x)
        return x_in + x

    def res_blocks(x, level):
        for _ in range(num_blocks):
            x = res_block(x, level)
        return x

    def down_level(x, level):
        # Residual conv blocks
        x = res_blocks(x, level)

        # Downsampling using strided conv
        return kl.Conv2D(filters[level + 1], kernel_size=2, strides=2,
                         use_bias=use_bias, padding='SAME')(x)

    def up_level(x, level):
        # Upsampling using tanspose conv
        x = kl.Conv2DTranspose(filters[level], kernel_size=2, strides=2,
                               use_bias=use_bias, padding='SAME')(x)

        # Residual conv blocks
        return res_blocks(x, level)

    # The complete model
    def call(x):
        # Pad to make sure the dimensions always fit
        x, (h, w) = pad_input(x)

        # Head
        x_head = kl.Conv2D(filters[0], kernel_size=3, use_bias=False, padding='SAME')(x)
        x = x_head

        # Downsample
        x_level = []
        for level in range(num_levels - 1):
            x = down_level(x, level)
            x_level.append(x)

        # Body
        x = res_blocks(x, num_levels - 1)

        # Upsample
        for level in range(num_levels-2, -1, -1):
            x = up_level(x + x_level[level], level)

        # Tail
        x = kl.Conv2D(out_channels, kernel_size=3,
                      use_bias=use_bias, padding='SAME')(x + x_head)

        # Crop back to the original size
        x = x[:, :h, :w, :]

        if final_clip:
            x = tf.clip_by_value(x, 0, 1)

        return x

    return call


def res_generator_model(in_channels=3, out_channels=3, num_residual_blocks=16):
    inp = kl.Input(shape=(None, None, in_channels))
    oup = res_generator(out_channels, num_residual_blocks)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def res_generator(out_channels=3, num_residual_blocks=16):
    encoder = blocks.conv2d_residual_encoder(
        num_residual_blocks=num_residual_blocks)
    out = kl.Conv2D(out_channels, kernel_size=(9, 9), padding='same')

    def call(x):
        x = encoder(x)
        x = out(x)
        x = tf.clip_by_value(x, 0, 1)
        return x

    return call


def rrdb_generator_model(in_channels=3, out_channels=3, filters=64, num_blocks=16,
                         inner_filters=32):
    inp = kl.Input((None, None, in_channels))
    oup = rrdb_generator(out_channels, filters, num_blocks, inner_filters)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def rrdb_generator(out_channels,
                   filters,
                   num_blocks,
                   inner_filters=32):
    lrelu = kl.LeakyReLU(0.2)

    conv_first = kl.Conv2D(filters, (3, 3), (1, 1), 'SAME', name='conv_first')

    rrdb_blocks = []
    for idx in range(num_blocks):
        rrdb_blocks.append(
            blocks.rrdb(filters, inner_filters,
                        activation=lrelu,
                        name='rrdb_' + str(idx)))

    conv_trunk = kl.Conv2D(filters, (3, 3), (1, 1), 'SAME', name='conv_trunk')

    # Final convolution layer
    conv_last = kl.Conv2D(out_channels, (3, 3), (1, 1), 'SAME', name='conv_last')

    def call(x):
        x = conv_first(x)
        x_trunk = x
        for b in rrdb_blocks:
            x_trunk = b(x_trunk)
        x_trunk = conv_trunk(x_trunk)
        x = x + x_trunk

        x = conv_last(x)
        x = tf.clip_by_value(x, 0, 1)
        return x

    return call
