# MIT License
#
# Copyright (c) 2018-2019 Michael Fuerst
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

import os
from os import listdir
from os.path import isfile, join
import sys
import json
import filecmp
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from setproctitle import setproctitle
from tensorflow.keras.utils import Sequence

from ailab import PHASE_TRAINVAL, PHASE_TEST, PHASE_TRAIN, PHASE_VALIDATION
from ailab.experiment import Config, import_config
from ailab.experiment import _log_code, _is_code_log_up_to_date

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_tf_record_pool_helper(args):
    config, dataset, num_threads, i, record_filename, iterator_mode = args
    thread_name = "%s:thread_%d" % (record_filename, i)
    _write_tf_record(config, dataset, num_threads, i, record_filename, thread_name=thread_name, iterator_mode=iterator_mode)


def _write_tf_record(config, dataset, num_threads, i, record_filename, thread_name="thread", iterator_mode=False):
    writer = tf.io.TFRecordWriter(record_filename)

    samples_written = 0
    if iterator_mode:
        chunk_size = int(config.problem.num_samples / config.train.batch_size)
    else:
        chunk_size = int(len(dataset) / num_threads)
    offset = i * chunk_size
    for idx in range(i, chunk_size):
        if iterator_mode:
            feature_batch, label_batch = next(dataset)
        else:
            feature_batch, label_batch = dataset[idx + offset]
        batch_size = list(feature_batch.values())[0].shape[0]
        for batch_idx in range(batch_size):
            feature_dict = {}

            for k in feature_batch.keys():
                feature_dict['feature_' +
                             k] = _bytes_feature(np.reshape(feature_batch[k][batch_idx], (-1,)).tobytes())
            for k in label_batch.keys():
                feature_dict['label_' + k] = _bytes_feature(np.reshape(label_batch[k][batch_idx], (-1,)).tobytes())

            example = tf.train.Example(features=tf.train.Features(
                feature=feature_dict))
            writer.write(example.SerializeToString())
            samples_written += 1
            if samples_written % 1000 == 0:
                print("Samples written by %s: %d." % (thread_name, samples_written))
    print("Samples written by %s: %d." % (thread_name, samples_written))
    writer.close()


def _read_tf_record(record_filename, data_config):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(record_filename)

    feature_dict = {}
    for k in data_config.keys():
        if "feature_" in k or "label_" in k:
            feature_dict[k] = tf.FixedLenFeature([], tf.string)

    data = tf.parse_single_example(
        serialized_example,
        features=feature_dict)

    outputs = {}
    for k in feature_dict.keys():
        feature_shape = data_config[k]["shape"]
        feature_type = np.dtype(data_config[k]["dtype"])
        feature = tf.decode_raw(data[k], feature_type)
        feature_len = 1
        for x in list(feature_shape):
            feature_len *= x
        feature.set_shape((feature_len,))
        outputs[k] = feature

    return outputs


def _create_parser_fn(data_config, phase):
    def parser_fn(serialized_example):
        tensor_dict = {}
        for k in data_config.keys():
            if "feature_" in k or "label_" in k:
                tensor_dict[k] = tf.io.FixedLenFeature([], tf.string)

        data = tf.io.parse_single_example(
            serialized_example,
            features=tensor_dict)

        outputs = {}
        for k in tensor_dict.keys():
            tensor_shape = data_config[k]["shape"]
            tensor_type = np.dtype(data_config[k]["dtype"])
            tensor = tf.io.decode_raw(data[k], tensor_type)
            tensor_len = 1
            for x in list(tensor_shape):
                tensor_len *= x
            tensor.set_shape((tensor_len,))
            outputs[k] = tensor

        features = {}
        labels = {}
        for k in outputs.keys():
            shape = tuple(list(data_config[k]["shape"]))
            tensor = tf.reshape(outputs[k], shape, name="input/" + phase + "/" + k + "_reshape")
            if "feature_" in k:
                features["_".join(k.split("_")[1:])] = tensor
            if "label_" in k:
                labels["_".join(k.split("_")[1:])] = tensor

        return features, labels
    return parser_fn


def _read_data(prefix, batch_size, augmentation=None, repeat=True, subset=None):
    """
    Loads a dataset.

    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :param augmentation: An augmentation function.
    :param subset: A list of indices of the record files to use, can be used for cross validation. If none is provided all tfrecord files are used.
    :return: A tensorflow.data.dataset object.
    """
    prefix = prefix.replace("\\", "/")
    folder = "/".join(prefix.split("/")[:-1])
    phase = prefix.split("/")[-1]
    data_config = json.load(open(prefix + '_config.json'))
    num_threads = data_config["num_threads"]

    filenames = [folder + "/" + f for f in listdir(folder) if isfile(join(folder, f))
                 and phase in f and not "config.json" in f]

    if subset is not None:
        filenames = filenames[subset]

    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=num_threads)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(map_func=_create_parser_fn(data_config, phase), num_parallel_calls=num_threads)
    if augmentation is not None:
        dataset = dataset.map(map_func=augmentation, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def create_input_fn(config, phase, augmentation_fn=None, repeat=True, subset=None):
    """
    Loads a dataset.

    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :param augmentation: An augmentation function.
    :param subset: A list of indices of the record files to use, can be used for cross validation. If none is provided all tfrecord files are used.
    :return: An input function for a tf estimator.
    """
    assert phase in [PHASE_TRAINVAL, PHASE_TEST, PHASE_TRAIN, PHASE_VALIDATION]

    prefix = os.path.join(config.problem.tf_records_path, phase)
    prefix = prefix.replace("\\", "/")
    folder = "/".join(prefix.split("/")[:-1])
    phase = prefix.split("/")[-1]
    data_config = json.load(open(prefix + '_config.json'))

    def input_fn():
        return _read_data(prefix, config.train.batch_size, augmentation_fn, repeat=repeat, subset=subset)
    return input_fn, data_config["num_samples"]


def write_data(config,
               phase,
               dataset,
               num_record_files=10,
               num_threads=1,
               multi_processing=False):
    """
    Write a tf record containing a feature dict and a label dict.

    :param config: The hyper parameters required for writing config.problem.tf_records_path: str. In case your sequence is an iterator and not a list interface config.problem.num_samples: int and config.train.batch_size: int is required
    :param mode: The mode specifies the purpose of the data. Typically it is either "train", "val", "trainval" or "test".
    :param dataset: Your actual data as a tf.keras.utils.sequence, a tf.data.Dataset, a generator or a list.
    :param num_record_files: The number of threads. If you use trainval mode 10 is nice because it gives you the ability to have even cross validation splits at 10% steps.
    :param num_threads: The number of threads uses for generating the record files. (Value is automatically cropped to 1 thread per record file.)
    :param multi_processing: If the processing is done with multiprocessing.Pool. (In case of generators and tf.data.Dataset this gets set to False automatically)
    :return:
    """
    
    if not os.path.exists(config.problem.tf_records_path):
        os.makedirs(config.problem.tf_records_path)
    outp_dir = os.path.join(config.problem.tf_records_path, "src")    
    if _is_code_log_up_to_date(outp_dir):
        _log_code(outp_dir)
    
    num_threads = max(num_threads, num_record_files)
    if isinstance(dataset, Sequence) or (callable(getattr(dataset, "__getitem__", None)) and callable(getattr(dataset, "__len__", None))):
        iterator_mode = False
    elif isinstance(dataset, tf.data.Dataset) or (callable(getattr(dataset, "__next__", None))):
        iterator_mode = True
        multi_processing = False
        num_threads = 1
    else:
        raise ValueError(
            "dataset must be tf.keras.utils.Sequence or a subtype or implement __len__(self) and __getitem__(self, idx) or the __next__(self) or be a subtype of tf.data.Dataset")

    if iterator_mode:
        if config.problem.get("num_samples", None) is None or config.train.get("batch_size", None) is None:
            raise RuntimeError("The config must specify  config.problem.num_samples and config.train.batch_size in iterator mode.")
    prefix = os.path.join(config.problem.tf_records_path, phase)
    prefix = prefix.replace("\\", "/")
    data_tmp_folder = "/".join(prefix.split("/")[:-1])
    if not os.path.exists(data_tmp_folder):
        os.makedirs(data_tmp_folder)

    args = [(config, dataset, num_record_files, i, (prefix + "_%d.tfrecords") % i, iterator_mode) for i in range(num_record_files)]

    # Retrieve a single batch
    if iterator_mode:
        # FIXME this example will not be in the final data I think
        sample_feature, sample_label = next(dataset)
    else:
        sample_feature, sample_label = dataset[0]

    data_config = {"num_threads": num_record_files, "num_samples": len(dataset)}
    for k in sample_feature.keys():
        data_config["feature_" + k] = {"shape": sample_feature[k].shape[1:], "dtype": sample_feature[k].dtype.name}
    for k in sample_label.keys():
        data_config["label_" + k] = {"shape": sample_label[k].shape[1:], "dtype": sample_label[k].dtype.name}

    with open(prefix + '_config.json', 'w') as outfile:
        json.dump(data_config, outfile)

    if iterator_mode or not multi_processing:
        for arg in args:
            _write_tf_record_pool_helper(arg)
    else:
        pool = Pool(processes=num_threads)
        pool.map(_write_tf_record_pool_helper, args)

    return True

def auto_setup_data(config, training_data=None, validation_data=None):
    if training_data is None and validation_data is None:
        augment_train = None
        augment_test = None
        if "augment" in config.arch.__dict__:
            augment = config.arch.augment()
            augment_train = augment.train
            augment_test = augment.test

        # Check if data is up to date
        needs_update = False
        if config.problem.get("tf_records_path", None) is not None:
            outp_dir = os.path.join(config.problem.tf_records_path, "src")
            if _is_code_log_up_to_date(outp_dir):
                needs_update = True

        # If data is not up to date or records should not be used load dataprovider
        if config.problem.get("tf_records_path", None) is None or needs_update:
            prepare = config.arch.prepare
            training_data = prepare(config, PHASE_TRAIN, augmentation_fn=augment_train)
            training_samples = len(training_data) * config.train.batch_size
            validation_data = prepare(config, PHASE_VALIDATION, augmentation_fn=augment_train)
            validation_samples = len(validation_data) * config.train.batch_size

        # Load the record dataset and update it if required.
        if config.problem.get("tf_records_path", None) is not None:  # Use tfrecords buffer
            tmp = config.train.batch_size
            config.train.batch_size = 1

            # When the training data is written, also update the validation data.
            if needs_update:
                write_data(config, PHASE_TRAIN, training_data)
                write_data(config, PHASE_VALIDATION, validation_data)
            config.train.batch_size = tmp

            training_data, training_samples = create_input_fn(
                config, PHASE_TRAIN, augmentation_fn=augment_train, repeat=False)
            training_data = training_data()
            validation_data, validation_samples = create_input_fn(
                config, PHASE_VALIDATION, augmentation_fn=augment_test, repeat=False)
            validation_data = validation_data()

    return training_data, training_samples, validation_data, validation_samples
