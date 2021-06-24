"""Blocks of layers to build models.
"""
import tensorflow as tf
from tensorflow.keras import layers as kl


def conv2d_block(filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu',
                 batch_normalization=True,
                 name='conv2d_block'):
    conv = kl.Conv2D(filters,
                     kernel_size,
                     strides,
                     padding,
                     name=name + '/conv2d')
    if batch_normalization:
        batch_norm = kl.BatchNormalization(name=name + '/bn')
    if callable(activation):
        activation = activation
    else:
        activation = kl.Activation(activation,
                                   name=name + '/activation')

    def call(x):
        x = conv(x)
        if batch_normalization:
            x = batch_norm(x)
        x = activation(x)
        return x

    return call


def conv2d_encoder(filter_base,
                   num_res_levels,
                   blocks_per_res,
                   batch_normalization='all',
                   name='conv2d_encoder',
                   **conv_kwargs):

    filters = [filter_base * 2**i for i in range(num_res_levels)]
    if batch_normalization == 'all':
        bn = True
        activate_bn = True
    elif batch_normalization == 'after-first':
        bn = False
        activate_bn = True
    elif batch_normalization == 'none':
        bn = False
        activate_bn = False
    else:
        raise ValueError("batch_normalization must be one of ['all', 'after-first', 'none'].")

    # Build the layers
    layers = []
    for f_idx, f in enumerate(filters):
        for idx in range(blocks_per_res - 1):
            lay = conv2d_block(f,
                               strides=(1, 1),
                               batch_normalization=bn,
                               name=name + '/block_' + str(f_idx) + '_' +
                               str(idx),
                               **conv_kwargs)
            layers.append(lay)
            bn = bn or activate_bn
        lay = conv2d_block(f,
                           strides=(2, 2),
                           batch_normalization=bn,
                           name=name + '/block_' + str(f_idx) + '_' +
                           str(blocks_per_res),
                           **conv_kwargs)
        layers.append(lay)
        bn = bn or activate_bn

    def call(x):
        for l in layers:
            x = l(x)
        return x

    return call


def residual_block(layers):
    def call(x):
        inp = x
        for l in layers:
            x = l(x)
        return x + inp

    return call


def conv2d_residual_block(filters,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          batch_normalization=True,
                          name='conv2d_residual'):
    return residual_block([
        conv2d_block(filters,
                     kernel_size,
                     strides,
                     padding,
                     activation,
                     batch_normalization,
                     name=name + '/conv_block_0'),
        conv2d_block(filters,
                     kernel_size,
                     strides,
                     padding,
                     'linear',
                     batch_normalization,
                     name=name + '/conv_block_1')
    ])


def conv2d_residual_encoder(num_residual_blocks=16,
                            activation=None,
                            name='conv2d_res_encoder'):
    if activation is None:
        activation = kl.PReLU(shared_axes=[1, 2])

    residual_blocks = []
    for idx in range(num_residual_blocks):
        residual_blocks.append(
            conv2d_residual_block(activation=activation,
                                  filters=64,
                                  kernel_size=(3, 3),
                                  name=name + '/res_block_' + str(idx)))

    layers = [
        conv2d_block(64,
                     kernel_size=(9, 9),
                     strides=(1, 1),
                     batch_normalization=False,
                     activation=activation,
                     name=name + '/conv2d_first'),
        residual_block(layers=[
            *residual_blocks,
            conv2d_block(filters=64,
                         kernel_size=(3, 3),
                         activation='linear',
                         name=name + '/conv2d_last')
        ]),
    ]

    def call(x):
        for l in layers:
            x = l(x)
        return x

    return call


def rrdb(filters=64,
         inner_filters=32,
         res_factor=0.2,
         num_blocks=3,
         name='rrdb',
         **rdb_kwargs):
    """Residual-in-Residual Dense Block
    """
    blocks = []
    for idx in range(num_blocks):
        blocks.append(
            residual_dense_block(filters, inner_filters,
                                 name=name + '/rdb_' + str(idx),
                                 **rdb_kwargs))

    def call(x):
        x_res = x
        for b in blocks:
            x_res = b(x_res)
        return x_res * res_factor + x

    return call


def residual_dense_block(
        filters=64,
        inner_filters=32,
        num_conv=5,
        res_factor=0.2,
        use_bias=True,
        activation='relu',
        name='rdb'):
    """Residual Dense Block
    """
    conv_layers = []
    filters_list = [inner_filters] * (num_conv - 1) + [filters]
    for idx, f in enumerate(filters_list):
        conv_layers.append(
            kl.Conv2D(f, (3, 3), (1, 1), 'SAME',
                      use_bias=use_bias,
                      name=name + '/conv_' + str(idx)))

    # Activation
    if callable(activation):
        activation = activation
    else:
        activation = kl.Activation(activation)

    def call(x):
        x_cat = x
        for l in conv_layers[:-1]:
            x_conv = activation(l(x_cat))
            x_cat = tf.concat([x_cat, x_conv], axis=-1)
        return conv_layers[-1](x_cat) * res_factor + x

    return call
