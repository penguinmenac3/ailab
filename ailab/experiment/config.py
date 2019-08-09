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
import os
from typing import Dict, Any
import json
from importlib import import_module
import inspect

import ailab

_sentinel = object()


class ConfigPart(object):
    """
    Converts a dictionary into an object.
    """

    def __init__(self, **kwargs):
        """
        Create an object from a dictionary.

        :param d: The dictionary to convert.
        """
        self.immutable = False
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict:
        dictionary = dict((key, value.to_dict()) if isinstance(value, ConfigPart) else (key, value)
                          for (key, value) in self.__dict__.items())
        del dictionary["immutable"]
        return dictionary

    def __repr__(self) -> str:
        return "ConfigPart(" + self.__str__() + ")"

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def get(self, key: str, default: Any = _sentinel) -> Any:
        """
        Get the value specified in the dictionary or a default.
        :param key: The key which should be retrieved.
        :param default: The default that is returned if the key is not set.
        :return: The value from the dict or the default.
        """
        if default is _sentinel:
            default = ConfigPart()
        return self.__dict__[key] if key in self.__dict__ else default

    def __getitem__(self, key: str) -> Any:
        """
        Get the value specified in the dictionary or a dummy.
        :param key: The key which should be retrieved.
        :return: The value from the dict or a dummy.
        """
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if "immutable" not in self.__dict__:  # In case users might not call constructor
            self.__dict__["immutable"] = False
        if self.immutable:
            raise RuntimeError("Trying to modify hyperparameters outside constructor.")

        if isinstance(value, str):
            # Try to match linux path style with anything that matches
            for env_var in list(os.environ.keys()):
                s = "$" + env_var
                value = value.replace(s, os.environ[env_var].replace("\\", "/"))

            # Try to match windows path style with anything that matches
            for env_var in list(os.environ.keys()):
                s = "%" + env_var + "%"
                value = value.replace(s, os.environ[env_var].replace("\\", "/"))

            if "%" in value or "$" in value:
                raise RuntimeError("Cannot resove all environment variables used in: '{}'".format(value))
        super.__setattr__(self, key, value)

    def __eq__(self, other: 'ConfigPart') -> bool:
        if not isinstance(other, ConfigPart):
            # don't attempt to compare against unrelated types
            return NotImplemented

        for k in self.__dict__:
            if not k in other.__dict__:
                return False
            if not self.__dict__[k] == other.__dict__[k]:
                return False

        for k in other.__dict__:
            if not k in self.__dict__:
                return False

        return True


class Config(ConfigPart):
    def __init__(self) -> None:
        """
        A configuration for a deep learning project.

        This class should never be instantiated directly, subclass it instead.

        The following parameters are set by default and should be changed after calling super.
        train.batch_size = 1
        train.experiment_name = None
        train.checkpoint_path = "checkpoints"
        train.epochs = 50
        train.log_steps = 100
        train.learning_rate.type = "const"
        train.learning_rate.start_value = 0.001
        train.learning_rate.end_value = 0.0001
        train.optimizer.type = "adam"
        arch.model = None
        arch.loss = None
        arch.metrics = None
        arch.prepare = None
        problem.base_dir = None

        You can add further attributes by simply adding them.
        When you want to add a new namespace like the existing train. You should use a ConfigPart() object for that.
        """
        self.train = ConfigPart()
        self.train.batch_size = 1
        self.train.experiment_name = None
        self.train.checkpoint_path = "checkpoints"
        self.train.epochs = 50
        self.train.log_steps = 100
        self.train.learning_rate = ConfigPart()
        self.train.learning_rate.type = "const"
        self.train.learning_rate.start_value = 0.001
        self.train.learning_rate.end_value = 0.0001
        self.train.optimizer = ConfigPart()
        self.train.optimizer.type = "adam"

        self.arch = ConfigPart()
        self.arch.model = None
        self.arch.loss = None
        self.arch.metrics = None
        self.arch.prepare = None

        self.problem = ConfigPart()
        self.problem.base_dir = None

        super().__init__()

    def __repr__(self) -> str:
        return "Config(" + self.__str__() + ")"

    @staticmethod
    def __has_attribute(obj: object, name: str) -> bool:
        """
        Checks if the object has an attribute.

        :param obj: The object that should be checked.
        :param name: The attribute that should be found.
        :return: True if the object has the attribute, False otherwise.
        """
        return name in obj.__dict__ and obj.__dict__[name] is not None

    def check_completness(self) -> bool:
        """
        Check the config for completeness.

        This method checks for the common bare minimum.
        If it fails to find something it throws an assertion error.
        :return: True if no exception occurs.
        """
        # Check for training parameters
        assert Config.__has_attribute(self, "train")
        assert Config.__has_attribute(self.train, "experiment_name")
        assert Config.__has_attribute(self.train, "checkpoint_path")
        assert Config.__has_attribute(self.train, "batch_size")
        assert Config.__has_attribute(self.train, "epochs")

        assert Config.__has_attribute(self.train, "optimizer")
        assert Config.__has_attribute(self.train.optimizer, "type")

        assert Config.__has_attribute(self.train, "learning_rate")
        assert Config.__has_attribute(self.train.learning_rate, "type")
        assert Config.__has_attribute(self.train.learning_rate, "start_value")
        if self.train.learning_rate.type == "exponential":
            assert Config.__has_attribute(self.train.learning_rate, "end_value")

        assert Config.__has_attribute(self, "arch")
        assert Config.__has_attribute(self.arch, "model")
        assert Config.__has_attribute(self.arch, "loss")
        assert Config.__has_attribute(self.arch, "metrics")
        assert Config.__has_attribute(self.arch, "prepare")

        assert Config.__has_attribute(self, "problem")
        assert Config.__has_attribute(self.problem, "base_dir")

        ailab.config = self
        return True

    @staticmethod
    def dynamic_import(obj: str) -> object:
        """
        Import an object dynamically by providing a string.

        :param obj: The string should specify the fully qualified module and class name.
        For example "ailab.experiment.config.Config" would specify a config. (structure: "module.submodule.Class")
        :return: The object that was loaded.
        """
        print("WARNING: Using this is highly discouraged. Import via normal python code instead.")
        p = ".".join(obj.split(".")[:-1])
        n = obj.split(".")[-1]
        module = __import__(p, fromlist=[n])
        return module.__dict__[n]


def import_config(config_file: str) -> Config:
    """
    Only libraries should use this method. Human users should directly import their configs.
    Automatically imports the most specific config from a given file.

    :param config_file: The configuration file which should be loaded.
    :return: The configuration object.
    """
    module_name = config_file.replace("\\", ".").replace("/", ".").replace(".py", "")
    module = import_module(module_name)
    symbols = list(module.__dict__.keys())
    symbols = [x for x in symbols if not x.startswith("__")]
    n = None
    for x in symbols:
        if not inspect.isclass(module.__dict__[x]):  # in Case we found something that is not a class ignore it.
            continue
        if issubclass(module.__dict__[x], Config):
            # Allow multiple derivatives of config, when they are derivable from each other in any direction.
            if n is not None and not issubclass(module.__dict__[x], module.__dict__[n]) and not issubclass(
                    module.__dict__[n], module.__dict__[x]):
                raise RuntimeError(
                    "You must only have one class derived from Config in {}. It cannot be decided which to use.".format(
                        config_file))
            # Pick the most specific one if they can be derived.
            if n is None or issubclass(module.__dict__[x], module.__dict__[n]):
                n = x
    if n is None:
        raise RuntimeError("There must be at least one class in {} derived from Config.".format(config_file))
    config = module.__dict__[n]()
    return config
