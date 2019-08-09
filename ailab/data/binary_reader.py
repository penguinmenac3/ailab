import struct
from typing import Callable, Any, List

_TYPE_REGISTRY = {}
_FMT_REGISTRY = {}


def _lenf(f):
    currentPos = f.tell()
    f.seek(0, 2)  # move to end of file
    length = f.tell()
    f.seek(currentPos, 0)
    return length


def _len_remaining(f):
    currentPos = f.tell()
    f.seek(0, 2)  # move to end of file
    length = f.tell()
    f.seek(currentPos, 0)
    return length - currentPos


def register_binary_fmt(type_char: str, fmt: str) -> Callable[[Callable], Callable]:
    """
    A class decorator to register a new datatype.
    The fmt must specify the types of the arguments of the class.
    To use the write feature a class must implement a serialize function, but for only reading it is not required.

    Example:

    >>> @register_binary_fmt(type_char="T", fmt="bb[e]e[s]e")
    >>> class ExampleType(object):
    >>> def __init__(self, i1: int, i2: int, arr1d: List[float], f1: float, s1: str, d1: float):
    >>>    self.i1 = i1
    >>>    self.i2 = i2
    >>>    self.arr1d = arr1d
    >>>    self.f1 = f1
    >>>    self.d1 = d1
    >>>    self.s1 = s1
    >>>
    >>> def serialize(self) -> [int, int, List[float], float, str, float]:
    >>>    return [self.i1, self.i2, self.arr1d, self.f1, self.s1, self.d1]

    All types that you registered and basic datatypes from struct are supported
    https://docs.python.org/3/library/struct.html
    Numbers like 4i are not supported, use iiii instead.

    1D Arrays are also supported by using [d] for a double array like [1,2,3].
    Arrays support any datatype. When using [s] it automatically gets joined to a string, since s is only one character of a string.

    :param type_char: The type character that should be assigned for this type. E.g. "T".
    :param fmt: The format of this type object. (The types of the constructor basically.)
    """

    def decorator_register_type(func: Callable) -> Callable:
        if type_char in _TYPE_REGISTRY:
            raise RuntimeError(
                "Each type can only be defined once. You already defined the type for {} somewhere else.".format(
                    type_char))
        _TYPE_REGISTRY[type_char] = func
        _FMT_REGISTRY[type_char] = fmt
        return func

    return decorator_register_type


def parse(fmt: str, binary_buffer, repeat: bool = False) -> Any:
    """
    All types that you registered and basic datatypes from struct are supported
    https://docs.python.org/3/library/struct.html
    Numbers like 4i are not supported, use iiii instead.

    1D Arrays are also supported by using [d] for a double array like [1,2,3].
    Arrays support any datatype. When using [s] it automatically gets joined to a string, since s is only one character of a string.

    :param fmt: The format string.
    :param binary_buffer: The binary buffer where to read from.
    :param repeat: If the format should be repeated if the buffer is not empty at the end of the format pattern.
    :return: The object(s) read from the buffer.
    """
    result = []
    while _len_remaining(binary_buffer) > 0:
        i = 0
        while i < len(fmt):
            c = fmt[i]
            # print(c)
            if c == "[":
                size = struct.calcsize("I")
                repeat_next = struct.unpack("I", binary_buffer.read(size))[0]
                index = fmt.find("]", i)
                loopy_part = fmt[i + 1:index]
                arr = []
                only_string = True
                for _ in range(repeat_next):
                    loopy = parse(loopy_part, binary_buffer)
                    arr.append(loopy)
                    if not isinstance(loopy, str):
                        only_string = False
                i = index
                c = fmt[i]
                if only_string:
                    arr = "".join(arr)
                result.append(arr)
            elif c not in _FMT_REGISTRY:
                size = struct.calcsize(c)
                data = struct.unpack(c, binary_buffer.read(size))[0]
                if c == "s":
                    data = data.decode("utf8")
                result.append(data)
            else:
                local_fmt = _FMT_REGISTRY[c]
                data = parse(local_fmt, binary_buffer)
                result.append(_TYPE_REGISTRY[c](*data))
            i += 1
        if not repeat:
            break

    if len(fmt) == 1 and not repeat:
        return result[0]
    return result


def write(fmt: str, obj: Any, binary_buffer) -> None:
    """
    All types that you registered and basic datatypes from struct are supported
    https://docs.python.org/3/library/struct.html
    Numbers like 4i are not supported, use iiii instead.

    1D Arrays are also supported by using [d] for a double array like [1,2,3].
    Arrays support any datatype. When using [s] it automatically gets joined to a string, since s is only one character of a string.

    :param fmt: The format string.
    :param obj: The object or list of objects that should be written. In case of a list the pattern will be repeated.
    :param binary_buffer: The binary buffer where to write to.
    """
    if not isinstance(obj, list) and not isinstance(obj, str):
        obj = [obj]
    i = 0
    for o in obj:
        c = fmt[i % len(fmt)]
        # print("{}: {}".format(c, o))
        if c == "[":
            index = fmt.find("]", i)
            binary_buffer.write(struct.pack("I", len(o)))
            loopy_part = fmt[i + 1:index]
            i = index
            write(loopy_part, o, binary_buffer)
        elif c not in _FMT_REGISTRY:
            if isinstance(o, str):
                o = o.encode("utf8")
            binary_buffer.write(struct.pack(c, o))
        else:
            write(_FMT_REGISTRY[c], o.serialize(), binary_buffer)
        i += 1
    # print("DONE")
    # print("*********************************")
