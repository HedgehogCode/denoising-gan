import tensorflow as tf
from tensorflow import keras

# Loss building blocks

cross_entropy = keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
mse = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
mae = keras.losses.MeanAbsoluteError(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
_vgg = None


def _get_vgg19():
    if _vgg is not None:
        return _vgg
    else:
        return keras.applications.VGG19(include_top=False, weights='imagenet')


def vgg19_loss(block_idx, conv_idx):
    mean = tf.constant([103.939, 116.779, 123.68])
    vgg = _get_vgg19()
    vgg_features = keras.models.Model(
        vgg.input,
        vgg.get_layer(f'block{block_idx}_conv{conv_idx}').output)

    def loss(y_true, y_pred):
        features_true = vgg_features((y_true * 255) - mean) / 12.75
        features_pred = vgg_features((y_pred * 255) - mean) / 12.75
        return mse(features_true, features_pred)

    return loss


def vgg19_ba_loss(block_idx, conv_idx):
    mean = tf.constant([103.939, 116.779, 123.68])
    vgg = _get_vgg19()

    layer_relu = vgg.get_layer(f'block{block_idx}_conv{conv_idx}')
    layer_idx = vgg.layers.index(layer_relu)
    layer_na = keras.layers.Conv2D(filters=layer_relu.filters,
                                   kernel_size=layer_relu.kernel_size,
                                   padding=layer_relu.padding)

    vgg_features = keras.models.Sequential([*vgg.layers[:layer_idx], layer_na])

    def loss(y_true, y_pred):
        features_true = vgg_features((y_true * 255) - mean) / 12.75
        features_pred = vgg_features((y_pred * 255) - mean) / 12.75
        return mse(features_true, features_pred)

    return loss


def disc_loss_clean(output_clean, output_gen):
    return cross_entropy(tf.ones_like(output_clean), output_clean)


def disc_loss_gen(output_clean, output_gen):
    return cross_entropy(tf.zeros_like(output_gen), output_gen)


def gen_loss_adv(_, __, ___, dis_oup):
    return cross_entropy(tf.ones_like(dis_oup), dis_oup)


def gen_loss_content(content_loss):
    def loss(real_inp, gen_oup, _, __):
        return content_loss(real_inp, gen_oup)
    return loss


# Combined losses for discriminator and generator

def disc_loss(output_clean, output_gen):
    return disc_loss_clean(output_clean, output_gen) + disc_loss_gen(output_clean, output_gen)


def gen_loss(content_loss, weight=1e-3):
    def loss(real_inp, gen_oup, dis_real_oup, dis_gen_oup):
        loss_adv = gen_loss_adv(real_inp, gen_oup, dis_real_oup, dis_gen_oup)
        loss_cont = content_loss(real_inp, gen_oup)
        return loss_cont + weight * loss_adv

    return loss


# Relativistic GAN loss

def _more_realistic(a, b):
    """A loss that drives a to be much bigger than b.
    Bigger means more realistic for the discriminator
    """
    Dab = a - tf.reduce_mean(b, axis=0)
    return cross_entropy(tf.ones_like(Dab), Dab)


def _less_realistic(a, b):
    """A loss that drives a to be much smaller than b.
    Smaller means less realistic for the discriminator
    """
    # A loss that drives a to be much bigger than b
    # Bigger means more realistic for the discriminator
    Dab = a - tf.reduce_mean(b, axis=0)
    return cross_entropy(tf.zeros_like(Dab), Dab)


def rel_disc_loss_morereal(output_clean, output_gen):
    return _more_realistic(output_clean, output_gen)


def rel_disc_loss_lessreal(output_clean, output_gen):
    return _less_realistic(output_gen, output_clean)


def rel_gen_loss_morereal(_, __, output_clean, output_gen):
    return _more_realistic(output_gen, output_clean)


def rel_gen_loss_lessreal(_, __, output_clean, output_gen):
    return _less_realistic(output_clean, output_gen)
