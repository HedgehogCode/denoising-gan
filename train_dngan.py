import os

import tensorflow as tf

import tensorflow_datasets_bw as datasets  # noqa: F401
from dngan import losses, training, metrics
import utils

# ============================================================================
# CONFIGURAION
# ============================================================================

DEBUG = 'DNGAN_DEBUG' in os.environ
if DEBUG:
    print('-----------------------------------')
    print('WARNING: Debug configuration active')
    print('-----------------------------------')

# Use values from env variables
logs_prefix = utils.get_from_environ('DNGAN_LOGS_PREFIX', 'logs')
checkpoints_prefix = utils.get_from_environ('DNGAN_CHECKPOINTS_PREFIX', 'checkpoints')
config_path = utils.get_from_environ('DNGAN_CONFIG', 'configs/drugan+_0.0-0.2.json')

# Configuration of the current GAN
config = utils.read_config(config_path)
config = utils.apply_debug(config, DEBUG)

# Configuration of the generator (initialized from a checkpoint)
config_generator_path = os.path.join(os.path.dirname(config_path), config['generator_config'])
config_generator = utils.read_config(config_generator_path)

if DEBUG:
    validation_per_step = 50
    checkpoints_per_step = 100
    test_per_step = 200
else:
    validation_per_step = 1000
    checkpoints_per_step = 10000
    test_per_step = 50000


# ============================================================================
# DATA
# ============================================================================

# Get the dataset
dataset_train, dataset_val = utils.get_train_val_datasets(config, DEBUG)
datasets_test = utils.get_test_datasets(config, debug=DEBUG)

# ============================================================================
# MODELS
# ============================================================================

# Define the models
generator = utils.get_generator(config_generator)
discriminator = utils.get_discriminator(config)

# ============================================================================
# TRAINING HELPERS
# ============================================================================

# Losses
discriminator_loss, discriminator_loss_weights = utils.get_discriminator_loss(config)
generator_loss, generator_loss_weights = utils.get_generator_loss(config)

# Optimizers
generator_optimizer, discriminator_optimizer = utils.get_optimiers(config)

# TensorBoard logging
tb_log_dir = os.path.join(logs_prefix, config['model_name'])

# Checkpoints
checkpoints_file_gen = os.path.join(checkpoints_prefix, config['model_name'], 'checkpoint_gen')
checkpoints_file_gan = os.path.join(checkpoints_prefix, config['model_name'], 'checkpoint_gan')

# Evaluation metrics
metrics_dict = {
    'psnr': metrics.psnr,
    'ssim': metrics.ssim,
    # 'sm-ssim': metrics.ms_ssim,
    'vgg22': losses.vgg19_loss(2, 2),
    'vgg54': losses.vgg19_loss(5, 4),
    'vgg22-ba': losses.vgg19_ba_loss(2, 2),
    'vgg54-ba': losses.vgg19_ba_loss(5, 4),
}

# ============================================================================
# LOAD PRETRAINED GENERATOR CHECKPOINT
# ============================================================================

utils.init_generator(config=config_generator,
                     checkpoints_prefix=checkpoints_prefix,
                     generator=generator,
                     optimizer=generator_optimizer,
                     checkpoints_per_step=checkpoints_per_step,
                     steps=config['steps_gen'])

# ============================================================================
# TRAINING PART 2 - GAN LOSS
# ============================================================================

trainer_gan = training.GANTrainer(generator, discriminator, checkpoints_file_gan, tb_log_dir)
trainer_gan.compile(gen_optimizer=generator_optimizer, dis_optimizer=discriminator_optimizer,
                    gen_loss=generator_loss, dis_loss=discriminator_loss,
                    gen_loss_weights=generator_loss_weights,
                    dis_loss_weights=discriminator_loss_weights,
                    metrics=metrics_dict)
trainer_gan.fit(data=dataset_train, steps=config['steps_gan'], initial_step=config['steps_gen'],
                validation_data=dataset_val,
                test_data=datasets_test,
                validation_per_step=validation_per_step,
                test_per_step=test_per_step,
                checkpoints_per_step=checkpoints_per_step,
                dis_per_step=config.get('dis_per_step', 1),
                gen_per_step=config.get('gen_per_step', 1))


# Final checkpoint
step = tf.Variable(config['steps_gan'], dtype=tf.int64)
checkpoint = trainer_gan._create_checkpoint(step)
checkpoint.save(file_prefix=checkpoints_file_gan)
