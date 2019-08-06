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


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    Extending on the pytorch dataset this dataset also needs to implement a ``version`` function.
    The version function returns a number (can be a hash) which changes, whenever the dataset changes.
    This enables subsequent callers to buffer this dataset and update their buffers when the version changes.
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def version(self):
        raise NotImplementedError


class Transformer(object):
    """
    A transformer should implement ``__call__``.
    """
    def __call__(self, args):
        raise NotImplementedError


class ComposeTransforms(Transformer):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, args):
        for t in self.transforms:
            args = t(args)
        return args


class TransformedDataset(object):
    def __init__(self, dataset, transformer):
        """
        Create a transfored dataset by applying a transformer.

        :param dataset: The dataset to transform.
        :param transformer: The transformer that gets applied to the dataset.
        """
        self.dataset = dataset
        self.transformer = transformer
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transformer(self.dataset[index])
