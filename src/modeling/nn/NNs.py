import tensorflow as tf
from tensorflow.keras import layers

from .CustomizedLayers import Slice


class Critic(tf.Module):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, num_elements: int,
                 kernel_size: int,
                 spectrum_dim: int = 2000,
                 spectrum_filters: int = 1, name=None):
        """_summary_

        Args:
            num_elements (int): _description_
            kernel_size (int): _description_
            spectrum_dim (int, optional): _description_. Defaults to 2000.
            spectrum_filters (int, optional): _description_. Defaults to 1.
            name (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(name=name)
        self.spectrum_filters = num_elements + spectrum_filters
        self.model = tf.keras.Sequential()
        self.model.add(layers.Input(
            shape=(spectrum_dim, self.spectrum_filters)))
        self.make_crit_block(kernel_size=kernel_size*1)
        self.make_crit_block(kernel_size=kernel_size*2)
        self.make_crit_block(kernel_size=kernel_size*4)
        self.make_final_block()

    def make_crit_block(self,
                        kernel_size: int, strides: int = 1,
                        padding: str = "same") -> None:
        """_summary_

        Args:
            kernel_size (int): _description_
            strides (int, optional): _description_. Defaults to 1.
            padding (str, optional): _description_. Defaults to "same".
        """
        self.model.add(layers.Conv1D(
            filters=self.spectrum_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        ))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(rate=0.2))
        self.model.add(layers.LeakyReLU(0.2))

    def make_final_block(self):
        """Creates the last block of layers"""
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(100))
        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation(None))


class Generator(tf.Module):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, z_dim: int,
                 hidden_dim: int,
                 num_elements: int,
                 kernel_size: int,
                 spectrum_dim: int = 2000,
                 spectrum_filters: int = 1, name=None):
        """[summary]

        Args:
            num_elements ([type]): [description]
            kernel_size (int): [description]
            spectrum_dim (int, optional): [description]. Defaults to 2000.
            spectrum_filters (int, optional): [description]. Defaults to 1.
            name ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        super().__init__(name=name)
        hidden_dim = 128
        im_dim = 2000
        shape = (num_elements + z_dim, spectrum_filters)
        self.spectrum_filters = spectrum_filters
        self.model = tf.keras.Sequential()
        self.model.add(layers.Input(shape=shape))
        self.model.add(layers.Reshape(shape[::-1]))

        self.model.add(layers.Conv1DTranspose(
            hidden_dim*4, kernel_size=24, strides=1))
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv1DTranspose(
            hidden_dim*2, kernel_size=50, strides=4))
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv1DTranspose(
            hidden_dim*1, kernel_size=50, strides=3))
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv1DTranspose(
            hidden_dim, kernel_size=50, strides=2))
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv1DTranspose(1, kernel_size=14, strides=2))

        self.model.add(layers.Activation("sigmoid"))
        self.model.add(layers.Reshape((im_dim, 1)))

    def make_final_block(self, spectrum_dim: int):
        """Creates the last block of layers"""
        self.model.add(layers.Activation("softmax"))
        self.model.add(layers.Reshape(target_shape=(2048, 1)))
        self.model.add(Slice(size=spectrum_dim))
