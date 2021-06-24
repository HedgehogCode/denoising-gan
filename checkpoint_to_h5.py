#!/usr/bin/env python
"""Save the latest checkpoint of a generator to a h5 Keras model
"""

from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

import utils


def main(args):
    # Config
    config = utils.read_config(args.config.name)

    if 'generator_config' in config:
        config_generator_path = os.path.join(os.path.dirname(args.config.name),
                                             config['generator_config'])
        config_generator = utils.read_config(config_generator_path)
    else:
        config_generator = config

    # Check if the config relates to a DnGAN or DAE
    if "discriminator_fn" in config:
        # Models
        print("Creating model...")
        generator = utils.get_generator(config_generator)
        discriminator = utils.get_discriminator(config)

        # Optimizers
        generator_optimizer, discriminator_optimizer = utils.get_optimiers(config)

        # Step
        step = tf.Variable(0, dtype=tf.int64)

        # Checkpoints
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
            step=step)
    else:
        # Models
        print("Creating model...")
        generator = utils.get_generator(config)

        # Optimizers
        optimizer, _ = utils.get_optimiers(config)

        # Step
        step = tf.Variable(0, dtype=tf.int64)

        # Checkpoints
        checkpoint = tf.train.Checkpoint(
            model=generator,
            optimizer=optimizer,
            step=step)

    # Restore
    checkpoints_dir = os.path.join(args.checkpoints, config['model_name'])
    checkpoint_file = tf.train.latest_checkpoint(checkpoints_dir)
    print(f"Restoring checkpoint \"{checkpoint_file}\"...")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir)).expect_partial()

    # Save as h5
    out_file = args.outfile.name
    print(f"Saving model to \"{out_file}\"...")
    generator.save(out_file)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config',
                        help="Model config file",
                        type=argparse.FileType('r'))
    parser.add_argument('outfile',
                        help="Output file",
                        type=argparse.FileType('w'))
    parser.add_argument('-c',
                        '--checkpoints',
                        help="Checkpoints directory",
                        default="checkpoints")

    return parser.parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(parse_args(sys.argv[1:])))
