import tensorflow as tf

from image_similarity_measures import quality_metrics


def psnr(imgs_a, imgs_b):
    return tf.math.reduce_mean(tf.image.psnr(imgs_a, imgs_b, max_val=1))


def ssim(imgs_a, imgs_b):
    return tf.math.reduce_mean(tf.image.ssim(imgs_a, imgs_b, max_val=1))


def ms_ssim(imgs_a, imgs_b):
    return tf.math.reduce_mean(tf.image.ssim_multiscale(imgs_a, imgs_b, max_val=1))


def fsim(imgs_a, imgs_b):
    def fsim_on_stacked_image(x):
        fsim_val = tf.numpy_function(quality_metrics.fsim, [x[..., 0], x[..., 1]], tf.float64)
        return tf.cast(fsim_val, tf.float32)
    stacked = tf.stack([imgs_a, imgs_b], axis=-1)
    fsim = tf.map_fn(fsim_on_stacked_image, stacked)  # type: ignore
    return tf.math.reduce_mean(fsim)
