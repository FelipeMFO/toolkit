# from ctypes.wintypes import PLARGE_INTEGER
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from src.modeling.nn.NNs import Critic, Generator
from src.modeling.nn.LossFunctions import LossFunctions
from src.visualization.Plots import Plots
from src.processing.Processing import Processing
from src.utils import split_list_into_batch_of_size_n


class WGAN(tf.keras.Model):
    """[summary]
    """

    def __init__(self, batch_size: int, crit: Critic, gen: Generator,
                 num_elements: int, num_epochs: int,
                 crit_repeats: int, gen_repeats: int, spectrum_dim: int,
                 spectrum_filters: int, z_dim: int, z_channels: int) -> None:

        super(WGAN, self).__init__()
        self.crit = crit
        self.gen = gen
        self.plots = Plots()
        self.proc = Processing()
        self.loss_functions = LossFunctions()

        self.batch_size = batch_size
        self.num_elements = num_elements
        self.num_epochs = num_epochs
        self.crit_repeats = crit_repeats
        self.gen_repeats = gen_repeats
        self.spec_shape = (spectrum_dim, spectrum_filters)
        self.z_shape = (z_dim, z_channels)

    def compile(self, lr):
        super(WGAN, self).compile()
        self.gen_opt = tf.keras.optimizers.RMSprop(lr)
        self.crit_opt = tf.keras.optimizers.RMSprop(lr)
        self.crit_losses, self.crit_grads, self.gen_losses, self.gen_grads = \
            [], [], [], []

    def get_noise(self, batch_size: int) -> EagerTensor:
        return tf.random.normal((batch_size, *self.z_shape))

    def add_one_hot_labels(self, unlabeled_tensor: EagerTensor or np.ndarray,
                           one_hot_labels: np.ndarray) -> EagerTensor:
        """_summary_

        Args:
            unlabeled_tensor (EagerTensor or np.ndarray): _description_
            one_hot_labels (np.ndarray): _description_

        Returns:
            EagerTensor: _description_
        """
        # Batch size verification
        assert len(unlabeled_tensor) == len(one_hot_labels)
        one_hot_tensor = tf.repeat(
            one_hot_labels.reshape(
                len(unlabeled_tensor), -1, self.num_elements),
            unlabeled_tensor.shape[1], axis=1)
        unlabeled_tensor = tf.cast(unlabeled_tensor, tf.float32)
        one_hot_tensor = tf.cast(one_hot_tensor, tf.float32)

        ans = tf.concat((unlabeled_tensor, one_hot_tensor), axis=2)
        return ans

    def train_step(self, dataset_batch: np.ndarray):
        """[summary]

        Args:
            real_spec (object): [description]

        Returns:
            [type]: [description]
        """
        one_hot_batch = dataset_batch[:, -1, self.spec_shape[1]:]
        batch_size = len(dataset_batch)

        real_spec_batch = self.add_one_hot_labels(
            one_hot_labels=one_hot_batch,
            unlabeled_tensor=dataset_batch[:, :, :self.spec_shape[1]])

        # Critic
        for _ in range(self.crit_repeats):
            for element_id in range(self.num_elements):
                noise = tf.concat(
                    (
                        self.get_noise(batch_size),
                        one_hot_batch.reshape(*one_hot_batch.shape, 1)
                    ),
                    axis=1)

                with tf.GradientTape() as crit_tape:
                    fake_spec = self.gen.model(noise, training=True)
                    fake_spec_one_hot = self.add_one_hot_labels(
                        fake_spec, one_hot_batch
                    )
                    real_pred = self.crit.model(real_spec_batch, training=True)

                    fake_pred = self.crit.model(
                        fake_spec_one_hot, training=True)
                    crit_loss = self.loss_functions.critic_loss(
                        crit_model=self.crit.model,
                        real_pred=real_pred,
                        fake_pred=fake_pred,
                        real_spec=real_spec_batch,
                        fake_spec=fake_spec_one_hot)

                crit_grad = crit_tape.gradient(
                    crit_loss, self.crit.trainable_variables)
                self.crit_opt.apply_gradients(
                    zip(crit_grad, self.crit.trainable_variables))

        # Generator
        for _ in range(self.gen_repeats):
            for element_id in range(self.num_elements):
                noise = tf.concat(
                    (
                        self.get_noise(batch_size),
                        one_hot_batch.reshape(*one_hot_batch.shape, 1)
                    ),
                    axis=1)

                with tf.GradientTape() as gen_tape:
                    gen_spec = self.gen.model(noise, training=True)
                    gen_spec_one_hot = self.add_one_hot_labels(
                        gen_spec, one_hot_batch
                    )
                    gen_pred = self.crit.model(gen_spec_one_hot, training=True)
                    gen_loss = self.loss_functions.generator_loss(gen_pred)

                gen_grad = gen_tape.gradient(gen_loss,
                                             self.gen.trainable_variables)
                self.gen_opt.apply_gradients(
                    zip(gen_grad, self.gen.trainable_variables))

    def train(self, dataset: np.ndarray, models_path: str, model_label: str):
        """Receives the dataset and launch train.
        dataset_split will be a list of numpy arrays with size of
        each batch_size

        Args:
            dataset (np.ndarray): _description_
            models_path (str): _description_
            model_label (str): _description_
        """
        real_examples = self.proc.get_spectrum_from_each_element(dataset)

        for epoch in range(self.num_epochs):
            start = time.time()

            # Batch split
            dataset_split = split_list_into_batch_of_size_n(
                dataset=dataset,
                batch_size=self.batch_size
            )

            # Train
            for batch in dataset_split:
                self.train_step(dataset_batch=batch)

            # Plots
            if epoch % 10 == 0:
                gen_specs = self.get_gen_specs_from_each_element()
                self.plots.plot_normalized_spectrums(gen_specs)
                self.plots.plot_normalized_spectrums(real_examples)
            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time()-start))

            # Checkpoints
            if epoch % 100 == 0:
                self.crit.model.save(
                    f"{models_path}{model_label}-critic_epoch-{epoch}.hdf5")
                self.gen.model.save(
                    f"{models_path}{model_label}-generator_epoch-{epoch}.hdf5")

    def get_gen_specs_from_each_element(self) -> list:
        """Uses the model generator to generate one spectrum
        from each element from random noises generated

        Returns:
            list: _description_
        """
        zs = [
            tf.concat(
                [
                    self.get_noise(batch_size=1),
                    tf.one_hot(i, self.num_elements).numpy().reshape(1, -1, 1)
                ],
                axis=1)
            for i in range(self.num_elements)
        ]
        gen_specs = [
            self.gen.model(z).numpy()[:, :, 0:self.spec_shape[1]]
            for z in zs
        ]
        return gen_specs
