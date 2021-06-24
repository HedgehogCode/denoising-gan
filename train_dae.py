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
checkpoints_prefix = utils.get_from_environ('DNGAN_CHECKPOINTS_PREFIX',
                                            'checkpoints')
config_path = utils.get_from_environ('DNGAN_CONFIG',
                                     'configs/dcnn_0.05.json')

# Configuration of the current GAN
config = utils.read_config(config_path)
config = utils.apply_debug(config, DEBUG, True)

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
generator = utils.get_generator(config)

# ============================================================================
# TRAINING HELPERS
# ============================================================================

# Losses
model_loss = {
    'mse': losses.mse
}

# Optimizers
optimizer, _ = utils.get_optimiers(config)

# TensorBoard logging
tb_log_dir = os.path.join(logs_prefix, config['model_name'])

# Checkpoints
checkpoints_file_gen = os.path.join(checkpoints_prefix, config['model_name'], 'checkpoint_gen')

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

lr_scheduler = utils.get_lr_scheduler(optimizer, config)

# ============================================================================
# TRAINING PART 1 - MSE LOSS
# ============================================================================

trainer_gen = training.ModelTrainer(generator, checkpoints_file_gen, tb_log_dir)
trainer_gen.compile(optimizer=optimizer, loss=model_loss, metrics=metrics_dict)
trainer_gen.fit(data=dataset_train, steps=config['steps_gen'], initial_step=0,
                validation_data=dataset_val,
                test_data=datasets_test,
                validation_per_step=validation_per_step,
                test_per_step=test_per_step,
                checkpoints_per_step=checkpoints_per_step,
                lr_scheduler=lr_scheduler)


# Final checkpoint
step = tf.Variable(config['steps_gen'], dtype=tf.int64)
checkpoint = trainer_gen._create_checkpoint(step)
checkpoint.save(file_prefix=checkpoints_file_gen)
