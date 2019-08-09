import os
import zipfile as zf
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple


class MultiZipReader(object):
    def __init__(self, *, zipfiles: List[str] = None, zipfile: str = None) -> None:
        """
        Create a zip dataset from either a single string or a list of strings pointing to zip files.

        :param zipfiles: A list of files to use for this dataset.
        """
        # Normalize parameters
        if zipfiles is None and zipfile is not None:
            zipfiles = [zipfile]
        elif zipfile is None and zipfiles is None:
            raise RuntimeError("You must either specify zipfile or zipfiles.")
        zipfiles = [os.path.normpath(x) for x in zipfiles]

        # Open zip files
        self.__zipfiles = [zf.ZipFile(filepath) for filepath in zipfiles]

        # Populate filelist and fileinfos
        self.__infos = []
        self.__filelist = []
        for i, z in enumerate(self.__zipfiles):
            zip_name = zipfiles[i].split(os.sep)[-1]
            data = {}
            for x in z.infolist():
                # TODO add merge mode where zips do not have their own namespace (usefull e.g. for loading kitti data)
                data[zip_name + os.sep + os.path.normpath(x.filename)] = x
                self.__filelist.append(zip_name + os.sep + os.path.normpath(x.filename))
            self.__infos.append(data)

        # Populate filetree
        self.__filetree = {}
        for filename in self.__filelist:
            tmp = filename.split(os.sep)
            folders = tmp[:-1]
            filename = tmp[-1]
            tmp = self.__filetree
            for f in folders:
                if f not in tmp:
                    tmp[f] = {}
                tmp = tmp[f]
            tmp[filename] = {}

    @property
    def file_list(self) -> List[str]:
        """
        The list of all files in all zip files.
        :return: A list containing all files.
        """
        return self.__filelist

    @property
    def file_tree(self) -> Dict:
        """
        The tree structure of all files in all zip files.
        :return:
        """
        return self.__filetree

    def __find_file(self, filename: str) -> Tuple[int, object]:
        """
        Find a file in all zips.
        :param filename: The filename to search.
        :return: In which zip it is findable with what fileinfo.
        """
        zipid = -1
        for i, inf in enumerate(self.__infos):
            if filename in inf.keys():
                zipid = i
                break
        if zipid < 0:
            raise RuntimeError("Cannot find file: ".format(filename))
        return zipid, self.__infos[zipid][filename]

    def read(self, filename: str) -> str:
        """
        Read a files content as a string.
        :param filename: The filename of the file.
        :return: The content of the file as a string.
        """
        zipid, fileinfo = self.__find_file(filename)
        with self.__zipfiles[zipid].open(fileinfo) as f:
            return f.read()

    def read_image(self, filename: str) -> np.ndarray:
        """
        Read an image from the zips.
        :param filename: The filename of the file.
        :return: The image as a numpy array.
        """
        zipid, fileinfo = self.__find_file(filename)
        with self.__zipfiles[zipid].open(fileinfo) as f:
            return np.array(Image.open(f))

    def open_file(self, filename: str) -> object:
        """
        Open a file with a file handle so user can decide what to do with it.
        :param filename: The file which to open.
        :return: The file handle of the file (like open(...)).
        """
        zipid, fileinfo = self.__find_file(filename)
        return self.__zipfiles[zipid].open(fileinfo)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    basedir = "T:\\Data\\Cityscapes"
    files = [os.path.join(basedir, x) for x in os.listdir(basedir) if x.endswith(".zip")]

    start = time.time()
    dataset = MultiZipReader(zipfiles=files)
    print(time.time() - start)

    print(len(dataset.file_list))
    fname = dataset.file_list[-1]
    print(fname)

    start = time.time()
    txt = dataset.read(fname)
    print(time.time() - start)

    images = [f for f in dataset.file_list if f.endswith(".png")]

    N = 1000
    start = time.time()
    image = None
    for img in images[-N:]:
        image = dataset.read_image(img)
    print(len(images[-N:]) / (time.time() - start))
    print(len(images[-N:]))

    plt.imshow(image)
    plt.show()
