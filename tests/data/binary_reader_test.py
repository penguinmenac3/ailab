from ailab.data.binary_reader import register_binary_fmt, parse, write


@register_binary_fmt(type_char="T", fmt="bb[e]e[s]e")
class TestType(object):
    def __init__(self, i1, i2, arr1d, f1, s1, d1):
        self.i1 = i1
        self.i2 = i2
        self.arr1d = arr1d
        self.f1 = f1
        self.d1 = d1
        self.s1 = s1

    def serialize(self):
        return [self.i1, self.i2, self.arr1d, self.f1, self.s1, self.d1]


if __name__ == "__main__":
    import struct

    arr1d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    obj = TestType(1, 2, arr1d, 3, "Hallo Welt", 4)
    original_objs = [obj for _ in range(10000)]

    # Write a list of objects with a specified format.
    print("Saving...")
    with open("tmp.bin", "wb") as fb:
        write("T", original_objs, fb)

    # Parse objects from a file using a format.
    print("Loading...")
    with open("tmp.bin", "rb") as fb:
        parsed_objs = parse("T", fb, repeat=True)

    assert len(parsed_objs) == len(original_objs)
    for a, b in zip(parsed_objs, original_objs):
        assert a.i1 == b.i1
        assert a.i2 == b.i2
        assert a.f1 == b.f1
        assert a.d1 == b.d1
        assert a.s1 == b.s1
        assert a.arr1d == b.arr1d

    print("Passed: Repeated writing of custom datatype.")
