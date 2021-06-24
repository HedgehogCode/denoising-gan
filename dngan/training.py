import abc
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class LRScheduler(abc.ABC):
    def __init__(self, optimizer: keras.optimizers.Optimizer,
                 log_name: str = 'lr'):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.log_name = log_name

    def update_lr(self, summary, step):
        lr = self._get_new_lr(summary, step)
        K.set_value(self.optimizer.lr, lr)
        summary[self.log_name] = lr
        return summary

    @abc.abstractmethod
    def _get_new_lr(self, summary, step):
        raise NotImplementedError(
            "Implement the _get_new_lr method in the LRScheduler implementation.")


class LRReduceOnPlateau(LRScheduler):
    def __init__(self,
                 optimizer: keras.optimizers.Optimizer,
                 monitor: str,
                 factor: float = 0.1,
                 patience: int = 10,
                 mode: str = 'min'):
        super(LRReduceOnPlateau, self).__init__(optimizer=optimizer)
        self.monitor = monitor
        self.factor = factor
        self.patience = patience

        self.wait = 0
        if mode == 'min':
            self.improved = lambda x: x < self.best
            self.best = tf.constant(np.inf)
        elif mode == 'max':
            self.improved = lambda x: x > self.best
            self.best = tf.constant(-np.inf)
        else:
            raise ValueError("Mode must be 'min' or 'max'.")

    def _get_new_lr(self, summary, step):
        current = summary[self.monitor]
        old_lr = float(K.get_value(self.optimizer.lr))

        if self.improved(current):
            # The loss improved
            self.best = current
            self.wait = 0
            return old_lr

        # The loss did not improve
        self.wait += 1
        if self.wait >= self.patience:
            # We waited long. Reduce the lr
            new_lr = self.factor * old_lr
            self.wait = 0
            return new_lr

        return old_lr


class Trainer(abc.ABC):
    def __init__(self, checkpoints_file: str, log_dir: str):
        """A trainer which can be trained with the Keras API.

        Args:
            checkpoints_file: Path to the checkpoints file
            log_dir: Directory for the tensorboard logging
        """
        super(Trainer, self).__init__()

        self.checkpoints_file = checkpoints_file
        self.log_dir = log_dir

    def compile(self, *args, **kwargs):
        """Configures the trainer for training."""
        self._compile(*args, **kwargs)
        self.compiled = True

    def fit(self,
            data: tf.data.Dataset,
            steps: int = 1000,
            initial_step: int = 0,
            validation_data=None,
            test_data: Dict[str, tf.data.Dataset] = None,
            validation_per_step: int = 1000,
            test_per_step: int = 10000,
            checkpoints_per_step: int = 10000,
            lr_scheduler: LRScheduler = None,
            **kwargs):
        if not self.compiled:
            raise ValueError("Call #compile() before calling #fit()")

        step = tf.Variable(initial_step, dtype=tf.int64)
        if test_data is None:
            test_data = {}

        checkpoint = self._create_checkpoint(step)

        # Prepare the tb writer
        self.tb_writer = tf.summary.create_file_writer(self.log_dir)

        for inp in data.take(steps - initial_step):
            # Apply the train step
            summary = self._train_step(inp, step, **kwargs)
            self._log(summary, step)

            # Run the validation
            if step % validation_per_step == 0:
                val_summary = self._val_summary(validation_data)

                # Update the learning rate
                if lr_scheduler is not None:
                    val_summary = lr_scheduler.update_lr(val_summary, step)

                # Logging
                tf.print('Step:', step, ', Summary:', summary,
                         ', Validation summary:', val_summary)
                self._log(val_summary, step)

            # Run on test data
            if step > 0 and step % test_per_step == 0:
                test_summary, test_images = self._test_summary(test_data)

                # Logging
                tf.print('Step:', step, ', Test summary:', test_summary)
                self._log(test_summary, step)
                self._log_images(test_images, step)

            # Save a checkpoint
            if step > 0 and step % checkpoints_per_step == 0:
                checkpoint.save(file_prefix=self.checkpoints_file)

            # Update step
            step.assign_add(1)

    # Internal helper methods

    def _log(self, summary, step):
        with self.tb_writer.as_default():
            for k, v in summary.items():
                tf.summary.scalar(k, v, step=step)

    def _log_images(self, summary, step):
        with self.tb_writer.as_default():
            for k, v in summary.items():
                tf.summary.image(k, v, step=step)

    # Abstract methods

    @abc.abstractmethod
    def _compile(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Implement the _compile method in the Trainer implementation.")

    @abc.abstractmethod
    def _train_step(self, inp) -> Dict[str, tf.Tensor]:
        raise NotImplementedError(
            "Implement the _train_step method in the Trainer implementation.")

    @abc.abstractmethod
    def _create_checkpoint(self, step: tf.Tensor) -> tf.train.Checkpoint:
        raise NotImplementedError(
            "Implement the _create_checkpoint method" +
            " in the Trainer implementation.")

    @abc.abstractmethod
    def _val_summary(self,
                     validation_data: tf.data.Dataset) -> Dict[str, tf.Tensor]:
        # The trainer does not support a validation summary
        return {}

    @abc.abstractmethod
    def _test_summary(self,
                      test_data: Dict[str, tf.data.Dataset]) -> Dict[str, tf.Tensor]:
        # The trainer does not support a test summary
        return {}


class GANTrainer(Trainer):
    def __init__(self, gen: keras.Model, dis: keras.Model,
                 checkpoints_file: str, log_dir: str) -> None:
        """A GAN trainer which can be trained with the Keras API.

        Args:
            gen: The generator model
            dis: The discriminator model
        """
        super(GANTrainer, self).__init__(
            checkpoints_file=checkpoints_file, log_dir=log_dir)
        self.gen = gen
        self.dis = dis

    def _compile(self,
                 gen_optimizer: keras.optimizers.Optimizer,
                 dis_optimizer: keras.optimizers.Optimizer,
                 gen_loss: Dict[str, Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]],
                 dis_loss: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
                 gen_loss_weights: Dict[str, float] = None,
                 dis_loss_weights: Dict[str, float] = None,
                 metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None) -> None:
        """Configures the trainer for training.

        Args:
            gen_optimizer: The tf.keras.optimizers optimizer for the generator.
            dis_optimizer: The tf.keras.optimizers optimizer for the generator.
            gen_loss: A dictionary with the losses for the generator. The sum of the losses is
                considered.
            dis_loss: A dictionary with the losses for the generator. The sum of the losses is
                considered.
            gen_loss_weights: Weights of the generator losses.
            dis_loss_weights: Weights of the discriminator losses.
        """
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

        self.gen_loss_weights = _loss_weights(gen_loss_weights, gen_loss)
        self.dis_loss_weights = _loss_weights(dis_loss_weights, dis_loss)

        if metrics is None:
            self.metrics = {}
        else:
            self.metrics = metrics

    @tf.function
    def _train_step(self, inp, step, dis_per_step=1, gen_per_step=1) -> Dict[str, tf.Tensor]:
        gen_inp, real_inp = inp

        summary = {}

        # Tape the gradients
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # Run the generator
            gen_oup = self.gen(gen_inp, training=True)

            # Run the discriminator
            dis_real_oup = self.dis(real_inp, training=True)
            dis_gen_oup = self.dis(gen_oup, training=True)

            # Compute the loss for generator
            loss_gen = tf.constant(0, tf.float32)
            for n, l in self.gen_loss.items():
                val = l(real_inp, gen_oup, dis_real_oup, dis_gen_oup)
                summary['train/generator/' + n] = val
                loss_gen += self.gen_loss_weights[n] * val

            # Compute the loss for the discriminator
            loss_dis = tf.constant(0, tf.float32)
            for n, l in self.dis_loss.items():
                val = l(dis_real_oup, dis_gen_oup)
                summary['train/discriminator/' + n] = val
                loss_dis += self.dis_loss_weights[n] * val

        # Get the gradients
        gen_grad = gen_tape.gradient(loss_gen, self.gen.trainable_variables)
        dis_grad = dis_tape.gradient(loss_dis, self.dis.trainable_variables)

        # Apply the gradinets
        if step % gen_per_step == 0:
            self.gen_optimizer.apply_gradients(
                zip(gen_grad, self.gen.trainable_variables))
        if step % dis_per_step == 0:
            self.dis_optimizer.apply_gradients(
                zip(dis_grad, self.dis.trainable_variables))

        # Return the loss
        summary['train/discriminator/loss'] = loss_dis
        summary['train/generator/loss'] = loss_gen
        return summary

    def _create_checkpoint(self, step: tf.Tensor) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.dis_optimizer,
            generator=self.gen,
            discriminator=self.dis,
            step=step)

    def _val_summary(self, validation_data: tf.data.Dataset) -> Dict[str, tf.Tensor]:
        return evaluate(self._generator_fn, validation_data, self.metrics)

    def _test_summary(self, test_data: Dict[str, tf.data.Dataset]) -> Dict[str, tf.Tensor]:
        # TODO(benjamin) return images
        return evaluate_test(self._generator_fn, test_data, self.metrics)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def _generator_fn(self, x: tf.Tensor) -> tf.Tensor:
        return self.gen(x)


class ModelTrainer(Trainer):
    def __init__(self, model: keras.Model,
                 checkpoints_file: str, log_dir: str) -> None:
        """A model trainer which can be trained with the Keras API.

        Args:
            model: The model
        """
        super(ModelTrainer, self).__init__(
            checkpoints_file=checkpoints_file, log_dir=log_dir)
        self.model = model

    def _compile(self,
                 optimizer: keras.optimizers.Optimizer,
                 loss: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
                 loss_weights: Dict[str, float] = None,
                 metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None) -> None:
        """Configures the trainer for training.

        Args:
            optimizer: The tf.keras.optimizers optimizer for the model.
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = _loss_weights(loss_weights, loss)

        if metrics is None:
            self.metrics = {}
        else:
            self.metrics = metrics

    @tf.function
    def _train_step(self, data, step) -> Dict[str, tf.Tensor]:
        inp, oup = data

        summary = {}

        # Tape the gradients
        with tf.GradientTape() as grad_tape:
            # Run the model
            model_oup = self.model(inp, training=True)

            # Compute the loss
            loss = tf.constant(0, tf.float32)
            for n, l in self.loss.items():
                val = l(oup, model_oup)
                summary['train/' + n] = val
                loss += self.loss_weights[n] * val

        # Get the gradients
        grad = grad_tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradinets
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        # Return the loss
        summary['train/loss'] = loss
        return summary

    def _create_checkpoint(self, step: tf.Tensor) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=step)

    def _val_summary(self, validation_data: tf.data.Dataset) -> Dict[str, tf.Tensor]:
        return evaluate(self._model_fn, validation_data, self.metrics)

    def _test_summary(self, test_data: Dict[str, tf.data.Dataset]) -> Dict[str, tf.Tensor]:
        # TODO(benjamin) return images
        return evaluate_test(self._model_fn, test_data, self.metrics)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def _model_fn(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)


def evaluate(model: Callable[[tf.Tensor], tf.Tensor], data: tf.data.Dataset,
             metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
             summary_prefix: str = 'val',
             return_image: bool = False) -> Dict[str, tf.Tensor]:
    """Evaluate the model on the given data with the given metrics.

    Args:
        model: The model.
        data: A tf.data.Dataset with the data in the form (inp, oup).
        metrics: The metrics that should be evaluated.
        summary_prefix: A prefix for the summary name.
        return_image: If a summary with the first image should be returned

    Returns:
        A summary with the mean value of all metrics. If return_image is True
        a tuple will be returned and the second element is the image summary.
    """
    summary = {}
    image_summary = {}
    count = tf.constant(0, tf.int32)

    def summary_name(name):
        return summary_prefix + '/' + name

    # Initialize with 0
    for name in metrics:
        summary[summary_name(name)] = tf.constant(0, tf.float32)

    # Loop over the validation data and sum up metric
    for inp, oup in data:
        model_oup = model(inp)
        if return_image and count == 0:
            image_summary[summary_name('inp')] = inp
            image_summary[summary_name('oup')] = model_oup
            image_summary[summary_name('gt')] = oup
        for name, metric in metrics.items():
            v = tf.reduce_sum(metric(oup, model_oup))
            summary[summary_name(name)] += v
        count += 1

    # Compute the mean of each metric
    for name in metrics:
        summary[summary_name(name)] /= tf.cast(count, tf.float32)

    if return_image:
        return summary, image_summary
    return summary


def evaluate_test(model: Callable[[tf.Tensor], tf.Tensor],
                  datasets: Dict[str, tf.data.Dataset],
                  metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
                  summary_prefix: str = 'test') -> Tuple[Dict[str, tf.Tensor],
                                                         Dict[str, tf.Tensor]]:
    summary = {}
    for n, d in datasets.items():
        d = d.batch(1)
        prefix = f'{summary_prefix}/{n}'
        summ = evaluate(model=model, data=d, metrics=metrics,
                        summary_prefix=prefix, return_image=False)
        summary.update(summ)
    return summary, {}


def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    weights = {n: 1.0 for n in loss}
    if loss_weights is not None:
        weights.update(loss_weights)
    return weights
