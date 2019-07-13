# MIT License
#
# Copyright (c) 2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import ceil
import numpy as np


class DataProvider(object):
    """
    This simple sequence makes implementing keras sequences in the correct way far simpler.

    A subclass must implement a __num_samples() -> int method
    and a __get_sample(idx: int) -> (feature_dict, label_dict) method which returns a single sample.

    This class automatically then applies augmentation, then preparation and finally batches as specified in
    hyperparams.train.batch_size.
    """

    def __init__(self, config, phase):
        super().__init__()
        self.config = config
        self.phase = phase

    def __len__(self):
        raise NotImplementedError(
            "A subclass must implement this function to find out how many training samples it has.")

    def __getitem__(self, index):
        raise NotImplementedError(
            "A subclass must implement this. Returns a tuple of (feature, label) representing a single training sample.")

    def to_batched_data_provider(self):
        return BatchedDataProvider(self.config, self)

class BatchedDataProvider(object):
    def __init__(self, config, data_provider):
        """
        Make a batched data provider from any list object.

        :param config: Is an ailab.Config object.
        :param data_provider: Anything that implements the list interface (__len__ and __getitem__(idx)).
        """
        self.data_provider = data_provider
        self.config = config

    def __len__(self):
        return ceil(len(self.data_provider) / self.config.train.batch_size)

    def __getitem__(self, index):
        features = []
        labels = []
        batch_size = self.config.train.batch_size
        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.data_provider))):
            feature, label = self.data_provider[idx]
            features.append(feature)
            labels.append(label)

        # In case of a dict, keep it as a dict. (for multiple features, labels)
        if isinstance(features[0], dict):
            input_tensor_order = sorted(list(features[0].keys()))
            return {k: np.array([dic[k] for dic in features]) for k in input_tensor_order},\
                   {k: np.array([dic[k] for dic in labels]) for k in labels[0]}
        else:  # for single feature, label
            return np.array(features), np.array(labels)

class DataTransformer(object):
    def __init__(self, config, data_provider):
        """
        Make a batched data provider from any list object.

        :param config: Is an ailab.Config object.
        :param data_provider: Anything that implements the list interface (__len__ and __getitem__(idx)).
        """
        self.data_provider = data_provider
        self.config = config

    def transform_data(self, feature, label):
        raise NotImplementedError(
            "A subclass must implement this. Returns a tuple of (feature, label) representing a training sample.")

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, index):
        return self.transform_data(*self.data_provider[index])


class DataProviderPipeline(object):
    def __init__(self, *args):
        self.steps = list(args)

    def add_step(self, step):
        self.steps.append(step)

    def __call__(self, config, phase, augmentation_fn=None):
        tmp = None
        for step in self.steps:
            if tmp is None:
                tmp = step(config, phase)
            else:
                tmp = step(config, tmp)
        if augmentation_fn is not None:
            tmp = DataTransformer(config, tmp)
            tmp.transform_data = augmentation_fn
        return tmp
