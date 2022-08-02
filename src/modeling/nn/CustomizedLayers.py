from tensorflow.keras import layers


class Slice(layers.Layer):
    """Slice the previous layer 2nd dimension

    Args:
        layers (_type_): _description_
    """
    def __init__(self, size: int, **kwargs):
        """Initialize the layer

        Args:
            size (int): size of output
        """
        super(Slice, self).__init__(**kwargs)
        self.size = size

    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'begin': self.begin,
    #         'size': self.size,
    #     })
    #     return config

    def call(self, inputs):
        return inputs[:, :self.size, :]
