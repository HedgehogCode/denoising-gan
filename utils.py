import os
import json
from typing import Callable, Tuple

import tensorflow as tf
from tensorflow.python.types.core import Value
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import tensorflow_datasets_bw as datasets
from dngan import discriminator as discriminator_module
from dngan import generator as generator_module
from dngan import losses
from dngan import training


# =================================================================================================
# CONFIG HANDLING
# =================================================================================================

def get_from_environ(var_name, default_val):
    if var_name in os.environ:
        return os.environ[var_name]
    else:
        return default_val


def read_config(path_to_config):
    with open(path_to_config, 'rb') as json_file:
        config = json.load(json_file)
    return config


def apply_debug(config, debug, dae=False):
    if debug:
        config['steps_gen'] = 1000 if dae else 800
        config['steps_gan'] = 1800
        config['model_name'] = config['model_name'] + '__debug'
    return config


def include_noise_level_map(config):
    return 'degrade' in config and config['degrade']['type'] == 'gaussian-map'


# =================================================================================================
# DATASET HELPERS
# =================================================================================================

def get_train_val_datasets(config, debug=False):
    # Get the number of parallel calls from an env variable
    num_cpus = int(get_from_environ('DNGAN_NUM_CPUS', '-1'))  # NOTE: -1 means AUTOTUNE
    prefetch = int(get_from_environ('DNGAN_PREFETCH', '-1'))

    # Prepare and degrade function: Used for training and validation
    prepare = get_prepare_fn(config)
    degrade = get_degrade_fn(config)

    # The training dataset
    if 'dataset_name' in config:
        # Legacy
        dataset_train, dataset_train_info = tfds.load(config['dataset_name'], split='train',
                                                      with_info=True)
        dataset_train = dataset_train.map(datasets.get_value(
            config.get('dataset_image_key', 'image')))
        shuffle_buffer = 16 if debug else dataset_train_info.splits['train'].num_examples
    else:
        dataset_train = None
        for dataset_config in config['train_datasets']:
            ds = tfds.load(dataset_config['name'], split='train') \
                .map(datasets.get_value(dataset_config['image_key']))
            if dataset_train is None:
                dataset_train = ds
            else:
                dataset_train = dataset_train.concatenate(ds)

        shuffle_buffer = 16 if debug else config['shuffle_buffer']

    dataset_train = dataset_train \
        .cache() \
        .repeat() \
        .map(prepare, num_parallel_calls=num_cpus) \
        .map(degrade, num_parallel_calls=num_cpus) \
        .shuffle(shuffle_buffer) \
        .batch(config['batch_size']) \
        .prefetch(prefetch)

    # The validation dataset
    if 'dataset_name' in config:
        # Legacy key: dataset_name
        dataset_val = tfds.load(config['dataset_name'], split='validation') \
            .map(datasets.get_value(config.get('dataset_image_key', 'image')))
    else:
        # Use the new key: val_dataset
        dataset_val = tfds.load(config['val_dataset']['name'], split='validation') \
            .map(datasets.get_value(config['val_dataset']['image_key']))

    if debug:
        dataset_val = dataset_val.take(10)
    dataset_val = dataset_val \
        .cache() \
        .map(prepare, num_parallel_calls=num_cpus) \
        .map(degrade, num_parallel_calls=num_cpus) \
        .batch(config.get('val_batch_size', config['batch_size'])) \
        .prefetch(prefetch)

    return dataset_train, dataset_val


def get_prepare_fn(config):
    def prepare(x):
        x = datasets.to_float32(x)
        x = datasets.from_255_to_1_range(x)
        x_shape = tf.shape(x)
        if x_shape[0] <= config['img_size'][0] or x_shape[1] <= config[
                'img_size'][1]:
            x = datasets.resize(config['img_size'])(x)
        else:
            x = tf.image.random_crop(x, (*config['img_size'], 3))
        return x

    return prepare


def get_degrade_fn(config):
    # Legacy mode: Always gaussian with fixed stddev
    if 'degrade' not in config:
        return get_degrade_gaussian(lambda: config['noise_stddev'])

    degrade_config = config['degrade']

    # Gaussian noise
    if degrade_config['type'] == 'gaussian':
        # Constant stddev
        if isinstance(degrade_config['stddev'], float):
            def stddev():
                return degrade_config['stddev']
        # Draw stddev from a beta distribution
        else:
            beta = tfp.distributions.Beta(
                degrade_config['stddev']['alpha'], degrade_config['stddev']['beta'])

            def stddev():
                return beta.sample()

        return get_degrade_gaussian(stddev)
    elif degrade_config['type'] == 'gaussian-map':
        min_stddev, max_stddev = degrade_config['stddev']
        return get_degrade_gaussian_map(min_stddev=min_stddev, max_stddev=max_stddev)

    raise ValueError("Degrade config is invalid. Type must be 'gaussian'.")


def get_degrade_gaussian(
        stddev: Callable[[], int]) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    def degrade(x):
        noise = tf.random.normal(tf.shape(x), stddev=stddev())
        return x + noise, x
    return degrade


def get_degrade_gaussian_map(min_stddev, max_stddev):
    def degrade(x):
        stddev = tf.random.uniform([], minval=min_stddev, maxval=max_stddev)
        noise = tf.random.normal(tf.shape(x), stddev=stddev)
        noise_map = tf.broadcast_to(stddev, tf.shape(x)[:2])[..., None]
        return tf.concat([x + noise, noise_map], axis=-1), x
    return degrade


# Test datasets

def get_test_datasets(config, debug=False, dataset_filter=None, degrade_filter=None):
    test_datasets = {
        'bsds500': tfds.load('bsds500/dmsp', split='validation').map(test_prepare_fn('image')),
        'div2k': tfds.load('div2k', split='validation').map(test_prepare_fn('hr')),
        'set5': tfds.load('set5', split='test').map(test_prepare_fn('hr')),
        'set14': tfds.load('set14', split='test').map(test_prepare_fn('hr')),
    }
    if dataset_filter is not None:
        test_datasets = dict(filter(lambda e: dataset_filter(e[0]), test_datasets.items()))
    if debug:
        # Only take 10 images and resize
        test_datasets = {k: d.take(10).map(datasets.resize(config['img_size']))
                         for k, d in test_datasets.items()}
    return {
        f'{k_ds}/{k_dg}': ds.map(dg) for k_dg, dg in test_degrade_fns(config,
                                                                      degrade_filter).items()
        for k_ds, ds in test_datasets.items()
    }


def test_prepare_fn(key):
    get_image = datasets.get_value(key)

    def prepare(x):
        x = get_image(x)
        x = datasets.to_float32(x)
        return datasets.from_255_to_1_range(x)
    return prepare


# Degrade functions

def degrade_gauss_noise(noise_stddev, config):
    def degrade(x):
        noise = tf.random.normal(tf.shape(x), stddev=noise_stddev)
        if include_noise_level_map(config):
            noise_map = tf.broadcast_to(noise_stddev, tf.shape(x)[:2])[..., None]
            return tf.concat([x + noise, noise_map], axis=-1), x
        return x + noise, x
    return degrade


def degrade_salt_pepper_noise(ratio, config):
    def degrade(x):
        random = tf.random.uniform(tf.shape(x), minval=0, maxval=1)
        noisy = tf.where(random < ratio, tf.constant(1, tf.float32), x)
        noisy = tf.where(random < ratio / 2, tf.constant(0, tf.float32), noisy)
        if include_noise_level_map(config):
            # We just use the ratio and hope that this is a good tradeoff
            noise_map = tf.broadcast_to(ratio, tf.shape(x)[:2])[..., None]
            return tf.concat([noisy, noise_map], axis=-1), x
        return noisy, x
    return degrade


# TODO add more degrade functions
def test_degrade_fns(config, degrade_filter=None):
    degrade_fns = {
        'gauss_0.01': degrade_gauss_noise(0.01, config),
        'gauss_0.02': degrade_gauss_noise(0.02, config),
        'gauss_0.05': degrade_gauss_noise(0.05, config),
        'gauss_0.10': degrade_gauss_noise(0.10, config),
        'gauss_0.20': degrade_gauss_noise(0.20, config),
        'salt_pepper_0.01': degrade_salt_pepper_noise(0.01, config),
        'salt_pepper_0.02': degrade_salt_pepper_noise(0.02, config),
        'salt_pepper_0.05': degrade_salt_pepper_noise(0.05, config),
        'salt_pepper_0.10': degrade_salt_pepper_noise(0.10, config),
        'salt_pepper_0.20': degrade_salt_pepper_noise(0.20, config),
    }
    if degrade_filter is not None:
        return dict(filter(lambda e: degrade_filter(e[0]), degrade_fns.items()))
    return degrade_fns


# =================================================================================================
# MODEL HELPERS
# =================================================================================================

def get_models(config):
    return get_discriminator(config), get_generator(config)


def get_generator(config):
    generator_fn = getattr(generator_module, config['generator_fn'])
    in_channels = 4 if include_noise_level_map(config) else 3
    return generator_fn(in_channels, 3, **config['generator_kwargs'])


def get_discriminator(config):
    discriminator_fn = getattr(discriminator_module,
                               config['discriminator_fn'])
    return discriminator_fn((*config['img_size'], 3),
                            **config['discriminator_kwargs'])


def get_optimiers(config):
    generator_optimizer = tf.keras.optimizers.Adam(config.get('lr_gen', 1e-4))
    discriminator_optimizer = tf.keras.optimizers.Adam(config.get('lr_disc', 1e-4))
    return generator_optimizer, discriminator_optimizer


def get_lr_scheduler(optimizer, config):
    if 'lr_scheduler' in config:
        c = config['lr_scheduler']
        return {
            'LRReduceOnPlateau': training.LRReduceOnPlateau
        }[c['type']](optimizer, **c['kwargs'])

    return None


# =================================================================================================
# LOSSES
# =================================================================================================

def get_generator_loss(config):
    if 'generator_loss' not in config:
        # Use a simple mix of content and adverserial loss
        g_loss = {
            'content': get_content_loss(config),
            'adv': losses.gen_loss_adv
        }
        g_loss_weight = {
            'content': 1.0,
            'adv': config['adv_loss_weight']
        }
        return g_loss, g_loss_weight

    # Helper function to get the loss for a name
    def get_loss(c):
        name = c['name']

        # Special GAN losses
        if name == 'adv':
            return losses.gen_loss_adv
        if name == 'relativistic_more_real':
            return losses.rel_gen_loss_morereal
        if name == 'relativistic_less_real':
            return losses.rel_gen_loss_lessreal

        # Simple losses for the generator
        return losses.gen_loss_content({
            'mse': lambda: losses.mse,
            'mae': lambda: losses.mae,
            'vgg22': lambda: losses.vgg19_loss(2, 2),
            'vgg54': lambda: losses.vgg19_loss(5, 4),
            'vgg22_ba': lambda: losses.vgg19_ba_loss(2, 2),
            'vgg54_ba': lambda: losses.vgg19_ba_loss(5, 4),
        }[name]())

    # Combine the loss as defined in the config
    g_loss = {k: get_loss(c) for k, c in config['generator_loss'].items()}
    g_loss_weight = {k: c['weight'] for k, c in config['generator_loss'].items()}
    return g_loss, g_loss_weight


def get_content_loss(config):
    if 'content_loss' in config:
        loss = {
            'mse': lambda: losses.mse,
            'mae': lambda: losses.mae,
            'vgg22': lambda: losses.vgg19_loss(2, 2),
            'vgg54': lambda: losses.vgg19_loss(5, 4),
            'vgg22_ba': lambda: losses.vgg19_ba_loss(2, 2),
            'vgg54_ba': lambda: losses.vgg19_ba_loss(5, 4),
        }[config['content_loss']]()
    else:
        loss = losses.vgg19_loss(2, 2)
    return losses.gen_loss_content(loss)


def get_discriminator_loss(config):
    d_loss_name = config.get('discriminator_loss', 'gan')
    if d_loss_name == 'gan':
        # Default GAN Discriminator loss
        d_loss = {
            'gen_loss': losses.disc_loss_gen,
            'clean_loss': losses.disc_loss_clean
        }
        d_loss_weight = {
            'gen_loss': 1.0,
            'clean_loss': 1.0
        }
        return d_loss, d_loss_weight
    if d_loss_name == 'relativistic':
        # Relativistic discriminator loss
        d_loss = {
            'less_real': losses.rel_disc_loss_lessreal,
            'more_real': losses.rel_disc_loss_morereal
        }
        d_loss_weight = {
            'less_real': 1.0,
            'more_real': 1.0
        }
        return d_loss, d_loss_weight
    raise ValueError("'discriminator_loss' must be one of ['gan', 'relativistic']. " +
                     f" Got {d_loss_name}.")


# =================================================================================================
# LOADING CHECKPOINTS
# =================================================================================================

def init_generator(config, checkpoints_prefix, generator, optimizer, checkpoints_per_step, steps):
    step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        model=generator,
        optimizer=optimizer,
        step=step)
    checkpoints_dir = os.path.join(checkpoints_prefix, config['model_name'])
    index = int(steps / checkpoints_per_step)
    checkpoints_file = os.path.join(checkpoints_dir, f'checkpoint_gen-{index}')
    checkpoint.restore(checkpoints_file).expect_partial()
    if step.numpy() != steps:
        raise ValueError(f"No checkpoint for step {steps}.")
