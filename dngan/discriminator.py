from tensorflow import keras

from dngan import blocks


def srgan_discriminator_model(inp_shape):
    inp = keras.layers.Input(inp_shape)
    oup = srgan_discriminator()(inp)
    return keras.Model(inputs=inp, outputs=oup)


def srgan_discriminator():
    activation = keras.layers.LeakyReLU()
    encoder = blocks.conv2d_encoder(filter_base=64,
                                    num_res_levels=4,
                                    blocks_per_res=2,
                                    batch_normalization='after-first',
                                    activation=activation)
    dense = keras.layers.Dense(1024)
    out_dense = keras.layers.Dense(1)

    def call(x):
        x = encoder(x)
        x = keras.layers.Flatten()(x)
        x = dense(x)
        x = out_dense(x)
        return x

    return call


def convnet_discriminator_model(inp_shape, **kwargs):
    inp = keras.Input(inp_shape)
    oup = convnet_discriminator(**kwargs)(inp)
    return keras.Model(inputs=inp, outputs=oup)


def convnet_discriminator(filter_base=32, num_res_levels=4, conv_per_res=2, activation='relu',
                          kernel_size=3, batch_normalization=True, dense_layers=None,
                          downsampling='max-pool'):
    if dense_layers is None:
        dense_layers = [512, 512]

    if downsampling not in ['max-pool', 'avg-pool', 'strided']:
        raise ValueError("downsampling must be one of ['max-pool', 'avg-pool', 'strided']")

    def call(x):
        filters = filter_base
        for i in range(num_res_levels):
            for j in range(conv_per_res):
                # Set strides to 2 for the last layer in the resolution level
                if downsampling == 'strided' and j == conv_per_res - 1:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

                x = blocks.conv2d_block(filters, kernel_size=kernel_size, activation=activation,
                                        batch_normalization=batch_normalization,
                                        strides=strides, name=f'level{i}/conv{j}')(x)

            # Perform downsampling if not done by strided convolutions
            if downsampling == 'max-pool':
                x = keras.layers.MaxPooling2D(2, name=f'level{i}/maxpool')(x)
            elif downsampling == 'avg-pool':
                x = keras.layers.AveragePooling2D(2, name=f'level{i}/avgpool')(x)

            # Increase the number of filters
            filters *= 2

        if downsampling == 'strided':
            x = keras.layers.Flatten()(x)
        elif downsampling == 'max-pool':
            x = keras.layers.GlobalMaxPool2D()(x)
        else:
            x = keras.layers.GlobalAvgPool2D()(x)

        for dense in dense_layers:
            x = keras.layers.Dense(dense, activation=activation)(x)

        return keras.layers.Dense(1, activation='linear')(x)
    return call
