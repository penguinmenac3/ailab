import os
import shutil
import filecmp
import datetime
import time
import json
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from typing import Callable, List

import entangle
from ailab.experiment.config import Config

__log_file = None
__checkpoint_path = None
__last_progress = 0
__last_update = time.time()
__entanglement = None

PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", "checkpoints", "dist", "docs", "*.egg-info",
                      "tfrecords", "*.code-workspace", ".git"]


def __ignore(candidate: str, forbidden_list: List[str]) -> bool:
    # Parse list to find simple placeholder notations
    start_list = []
    end_list = []
    for item in forbidden_list:
        if item.startswith("*"):
            end_list.append(item.replace("*", ""))
        if item.endswith("*"):
            start_list.append(item.replace("*", ""))
    # Test
    res = candidate in forbidden_list
    for item in start_list:
        res |= candidate.startswith(item)
    for item in end_list:
        res |= candidate.endswith(item)
    return res


def __get_all_files(root: str, forbidden_list: List[str]) -> List[str]:
    all_files = []
    root_with_sep = root + os.sep
    for path, subdirs, files in os.walk(root):
        files = [x for x in files if not __ignore(x, forbidden_list)]
        subdirs[:] = [x for x in subdirs if not x.startswith(".") and not __ignore(x, forbidden_list)]
        for name in files:
            all_files.append(os.path.join(path, name).replace(root_with_sep, ""))
    return all_files


def _get_loaded_files(root: str = None, forbidden_list: List[str] = PYTHON_IGNORE_LIST) -> List[str]:
    """
    Get a list of all files that correspond to loaded modules in the root folder.

    If root is None the current cwd is used.
    """
    if root is None:
        root = os.getcwd()

    cwd_files = __get_all_files(root, forbidden_list)
    # TODO filter out all files that are not loaded.

    return cwd_files


def _get_backup_path(fname: str, outp_dir: str = None) -> str:
    assert outp_dir is not None

    return os.path.join(os.path.normpath(outp_dir), fname)


def _copyfile(src: str, dst: str, follow_symlinks: bool = True, create_missing_dirs: bool = True) -> None:
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    shutil.copyfile(src, dst, follow_symlinks=follow_symlinks)


def _write_log(*, obj: object) -> None:
    """
    Write a log to the logfile or console if none is available.
    Furthermore send it to the online server if it is connected.

    :param obj: The json serializable object to log.
    """
    global __log_file
    global __entanglement
    out_str = json.dumps(obj)
    if __log_file is None:
        print("WARING: You should setup logging before using it. Call ailab.logging.setup(...).")
        print(out_str)
    else:
        with open(__log_file, "a") as f:
            f.write(out_str + "\n")

    if __entanglement is not None:
        # TODO send to server
        pass


class LogResult(object):
    def __init__(self, *, name: str, primary: bool = False) -> None:
        """
        Annotation to log the result of a function.

        A simple example would be the primary loss the training pipeline.

        >>> @LogResult(name="loss", primary=True)
        >>> def loss(y_true, y_preds):
        >>>     return 42
        42

        :param name: The name for the logged result.
        :param primary: If the result is the primary loss.
        """
        self.name = name
        self.primary = primary

    def __call__(self, f: Callable) -> Callable:
        def wrapped_f(*args, **kwargs):
            result = f(*args, **kwargs)
            log_value(name=self.name, value=result, primary=self.primary)
            return result

        return wrapped_f


class LogCall(object):
    def __init__(self, *, name: str, primary: bool = False) -> None:
        """
        Annotation to log the number of times a function is called.

        The following example logs a variable step, which tracks how often the function step gets called.

        >>> @LogCall(name="step")
        >>> def step():
        >>>     # Fancy training code...
        >>>     return 42
        42

        :param name: The name of the logged function call counting.
        :param primary: If the logged result is the primary loss (it probably isn't)
        """
        self.name = name
        self.primary = primary
        self.i = 0

    def __call__(self, f: Callable) -> Callable:
        def wrapped_f(*args, **kwargs):
            log_value(name=self.name, value=self.i, primary=self.primary)
            self.i += 1
            result = f(*args, **kwargs)
            return result

        return wrapped_f


def log_value(*, name: str, value: object, primary: bool = False) -> None:
    """
    Log a value to the file or online server.
    :param name: The name of the value to be logged.
    :param value: The actual value. It can be anything that is json serializable.
    :param primary: When primary is true this is the metric that is displayed online in the results preview.
    It should be only true for the main loss.
    """
    date = {
        "timestamp": "{}".format(datetime.datetime.now()),
        "name": name,
        "value": value,
        "primary": primary
    }
    _write_log(obj=date)


def update_progress(progress: float) -> None:
    """
    Update the progress value. Automatically also computes the ETA and updates it in the logs.
    :param progress: A value between 0 and 1 indicating the progress, where 1 means done.
    The value should grow monotonic.
    """
    global __last_progress
    global __last_update

    assert 0 <= progress <= 1

    delta_t = time.time() - __last_update
    delta_p = max(progress - __last_progress, 1E-6)
    __last_update = time.time()
    __last_progress = progress
    eta = (1 - progress) / delta_p * delta_t

    date = {
        "timestamp": "{}".format(datetime.datetime.now()),
        "eta": int(eta),
        "progress": int(progress * 1000) / 1000
    }
    _write_log(obj=date)


def _log_code(*, chkpt_path: str, forbidden_list: list = []) -> None:
    """
    Log the code of the current working directory into the src folder of your checkpoint path.

    :param chkpt_path: The checkpoint folder.
    :param forbidden_list: The list of the forbidden files.
    """
    outp_dir = os.path.join(chkpt_path, "src")
    forbidden_list.extend(PYTHON_IGNORE_LIST)

    loaded_files = _get_loaded_files(forbidden_list=forbidden_list)
    # Copy preparation code to output location and load the module.
    for f in loaded_files:
        f_backup = _get_backup_path(f, outp_dir=outp_dir)
        _copyfile(f, f_backup)


def _is_code_log_up_to_date(*, chkpt_path: str, forbidden_list: list = []) -> bool:
    """
    Check if the code in the logs is up to date or needs updates.

    :param chkpt_path: The checkpoint folder.
    :param forbidden_list: The list of the forbidden files.
    :return: True if the files are up to date, False if not.
    """
    outp_dir = os.path.join(chkpt_path, "src")
    if not os.path.exists(outp_dir):
        return False
    forbidden_list.extend(PYTHON_IGNORE_LIST)
    loaded_files = _get_loaded_files(forbidden_list=forbidden_list)

    for f in loaded_files:
        f_backup = _get_backup_path(f, outp_dir=outp_dir)
        # Check if data is already up to date
        if not os.path.exists(f_backup) or not filecmp.cmp(f, f_backup):
            return False

    return True


def log_image(*, name: str, data: np.ndarray = None) -> None:
    """
    Log an image.
    :param name: The name of the image.
    :param data: The data (optional) if none is provided it is assumed that a pyplot figure should be saved.
    """
    global __checkpoint_path
    if __checkpoint_path is None:
        print("WARNING: Cannot log images when logging is not setup. Call logging.setup first")
        return
    if data is None:
        plt.savefig(os.path.join(__checkpoint_path, "images", name + ".png"))
    else:
        scipy.misc.imsave(os.path.join(__checkpoint_path, "images", name + ".png"), data)


def setup(config: Config, continue_with_specific_checkpointpath: bool = False, continue_training: bool = False) -> str:
    """
    Setup the logging.
    This creates the folder structure required at the place specified in config.train.checkpoint_path.
    After creating the folder structure it backs up the code of the current working directory to the folder structure.

    :param config: The configuration that is used for this run.
    :param continue_with_specific_checkpointpath: When a specific checkpoint is used to continue a run, set this.
    This avoids creating a new folder if it is not required.
    :param continue_training: Same as specific checkpoint but the checkpoint
    is automatically selected to be the most recent.
    This avoids creating a new folder if it is not required.
    :return: The path to the checkpoint folder.
    """
    global __log_file
    global __checkpoint_path
    if __log_file is not None:
        raise RuntimeError("You must not setup logging twice!")
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    chkpt_path = config.train.checkpoint_path + "/" + time_stamp
    chkpt_path = chkpt_path + "_" + config.train.experiment_name

    if continue_with_specific_checkpointpath:
        chkpt_path = config.train.checkpoint_path + "/" + continue_with_specific_checkpointpath
        print("Continue with checkpoint: {}".format(chkpt_path))
    elif continue_training:
        chkpts = sorted([name for name in os.listdir(config.train.checkpoint_path)])
        chkpt_path = config.train.checkpoint_path + "/" + chkpts[-1]
        print("Latest found checkpoint: {}".format(chkpt_path))

    if not os.path.exists(os.path.join(chkpt_path, "train")):
        os.makedirs(os.path.join(chkpt_path, "train"))
    if not os.path.exists(os.path.join(chkpt_path, "val")):
        os.makedirs(os.path.join(chkpt_path, "val"))
    if not os.path.exists(os.path.join(chkpt_path, "checkpoints")):
        os.makedirs(os.path.join(chkpt_path, "checkpoints"))
    if not os.path.exists(os.path.join(chkpt_path, "images")):
        os.makedirs(os.path.join(chkpt_path, "images"))

    if not _is_code_log_up_to_date(chkpt_path=chkpt_path):
        _log_code(chkpt_path=chkpt_path)
    __log_file = os.path.join(chkpt_path, "log.txt")
    __checkpoint_path = chkpt_path
    return chkpt_path


def connect(*, host: str, port: int, user: str, password: str) -> bool:
    """
    Connect to an ailab server for live logging of the experiment.

    :param host: Hostname of the server.
    :param port: The port on which ailab-server runs.
    :param user: Your username to authenticate in ailab.
    :param password: Your password to authenticate in ailab.
    :return: True if the connection was established False otherwise.
    """
    global __entanglement
    if __entanglement is not None:
        raise RuntimeError("You must not connect to a server for live logging twice!")
    __entanglement = entangle.connect(host=host, port=port, user=user, password=password)
    if __entanglement is None:
        return False

    def on_close():
        global __entanglement
        __entanglement = None
    __entanglement.on_close = on_close
    return True
