import datetime


log_file = "events.log"


class LogResult(object):
    def __init__(self, msg):
        self.msg = msg

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            result =  f(*args, **kwargs)
            log_event(self.msg.format(result))
            return result
        return wrapped_f


class LogCall(object):
    def __init__(self, msg):
        self.msg = msg

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            log_event(self.msg)
            result =  f(*args, **kwargs)
            return result
        return wrapped_f


def log_event(msg):
    out_str = "[{}] {}".format(datetime.datetime.now(), msg)
    with open(log_file, "a") as f:
        f.write(out_str + "\n")
    print(out_str)


if __name__ == "__main__":
    class Test(object):
        @LogCall("Test")
        @LogResult("Test: {}")
        def test(self, foobar):
            return foobar + " 42"


    t = Test()
    t.test("Hello")
