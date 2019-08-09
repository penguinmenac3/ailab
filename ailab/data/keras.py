import tensorflow as tf
from math import ceil
import numpy as np
import ailab
from typing import List


class BatchedKerasDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset: List) -> None:
        """
        Make a batched data provider from any list object.

        :param config: Is an ailab.Config object.
        :param dataset: Anything that implements the list interface (__len__ and __getitem__(idx)).
        """
        self.dataset = dataset
        self.config = ailab.config

    def __len__(self):
        return ceil(len(self.dataset) / self.config.train.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        batch_size = self.config.train.batch_size
        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.dataset))):
            feature, label = self.dataset[idx]
            features.append(feature)
            labels.append(label)

        # In case of a dict, keep it as a dict. (for multiple features, labels)
        if isinstance(features[0], dict):
            input_tensor_order = sorted(list(features[0].keys()))
            return {k: np.array([dic[k] for dic in features]) for k in input_tensor_order},\
                   {k: np.array([dic[k] for dic in labels]) for k in labels[0]}
        else:  # for single feature, label
            return np.array(features), np.array(labels)
