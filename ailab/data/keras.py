import tensorflow as tf

class Sequence(tf.keras.utils.Sequence):
    def __init__(self, config, data_provider):
        """
        Make a batched data provider from any list object.

        :param config: Is an ailab.Config object.
        :param data_provider: Anything that implements the list interface (__len__ and __getitem__(idx)).
        """
        self.data_provider = data_provider
        self.config = config

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, index):
        return self.data_provider[index]
