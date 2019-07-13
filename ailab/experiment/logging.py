import os
import shutil
import hashlib
import zipfile
import time
import datetime
import filecmp

PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", "checkpoints", "dist", "docs", "*.egg-info", "tfrecords", "*.code-workspace", ".git"]

def __ignore(candidate, forbidden_list):
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

def __get_all_files(root, forbidden_list):
    all_files = []
    root_with_sep = root + os.sep
    for path, subdirs, files in os.walk(root):
        files = [x for x in files if not __ignore(x, forbidden_list)]
        subdirs[:] = [x for x in subdirs if not x.startswith(".") and not __ignore(x, forbidden_list)]
        for name in files:
            all_files.append(os.path.join(path, name).replace(root_with_sep, ""))
    return all_files

def _get_loaded_files(root=None, forbidden_list=PYTHON_IGNORE_LIST):
    """
    Get a list of all files that correspond to loaded modules in the root folder.

    If root is None the current cwd is used.
    """
    if root is None:
        root = os.getcwd()

    cwd_files = __get_all_files(root, forbidden_list)
    # TODO filter out all files that are not loaded.

    return cwd_files


def _get_backup_path(fname, outp_dir=None):
    assert outp_dir is not None

    return os.path.join(os.path.normpath(outp_dir), fname)


def _copyfile(src, dst, follow_symlinks=True, create_missing_dirs=True):
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    shutil.copyfile(src, dst, follow_symlinks=True)

def backup(outp_dir, forbidden_list=[]):
    forbidden_list.extend(PYTHON_IGNORE_LIST)

    loaded_files = _get_loaded_files(forbidden_list=forbidden_list)
    # Copy preparation code to output location and load the module.
    for f in loaded_files:
        f_backup = _get_backup_path(f, outp_dir=outp_dir)
        _copyfile(f, f_backup)

def needs_backup(outp_dir, forbidden_list=[]):
    if not os.path.exists(outp_dir):
        return True
    forbidden_list.extend(PYTHON_IGNORE_LIST)
    loaded_files = _get_loaded_files(forbidden_list=forbidden_list)
    
    for f in loaded_files:
        f_backup = _get_backup_path(f, outp_dir=outp_dir)
        # Check if data is already up to date
        if not os.path.exists(f_backup) or not filecmp.cmp(f, f_backup):
            return True

    return False

def setup_logging(config, continue_with_specific_checkpointpath=False, continue_training=False):
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

    if not os.path.exists(chkpt_path + "/train"):
        os.makedirs(chkpt_path + "/train")
    if not os.path.exists(chkpt_path + "/val"):
        os.makedirs(chkpt_path + "/val")
    if not os.path.exists(chkpt_path + "/checkpoints"):
        os.makedirs(chkpt_path + "/checkpoints")
    if not os.path.exists(chkpt_path + "/logs"):
        os.makedirs(chkpt_path + "/logs")
    if not os.path.exists(chkpt_path + "/images"):
        os.makedirs(chkpt_path + "/images")

    backup(os.path.join(chkpt_path, "src"))
    return chkpt_path
