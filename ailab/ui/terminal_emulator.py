import json
import os
import shlex
import signal
import webbrowser
from pathlib import Path
from threading import Thread
WINDOWS = False
try:
    import pyte
    import pty
except:
    WINDOWS = True


class Terminal(object):
    def add_listener(self, listener):
        raise NotImplementedError()

    def remove_listener(self, listener):
        raise NotImplementedError()

    def resize(self, lines, columns):
        raise NotImplementedError()

    def feed(self, data):
        raise NotImplementedError()

    def dumps(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()


class WindowsTerminal(Terminal):
    def __init__(self, command, columns, lines, cwd=None):
        pass


class LinuxTerminal(Terminal):
    def __init__(self, command, columns, lines, cwd=None):
        pid, master_fd = pty.fork()
        if pid == 0:  # Child.
            if cwd is not None:
                os.chdir(cwd)
            argv = shlex.split(command)
            env = dict(TERM="linux", LC_ALL="en_GB.UTF-8",
                    COLUMNS=str(columns), LINES=str(lines),
                    HOME="/home/fuerst")
            os.execvpe(argv[0], argv, env)

        # File-like object for I/O with the child process aka command.
        file_handle = os.fdopen(master_fd, "w+b", 0)

        self.screen = pyte.Screen(columns, lines)
        self.screen.set_mode(pyte.modes.LNM)
        self.stream = pyte.ByteStream()
        self.stream.attach(self.screen)
        self.pid = pid
        self.pout = file_handle
        self.t = Thread(target=self.read_proc)
        self.t.start()
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def resize(self, lines, columns):
        self.screen.resize(lines, columns)

    def read_proc(self):
        while True:
            try:
                line = self.pout.read(1).decode("utf-8")
                if line == '':
                    break
                else:
                    self.stream.feed(line.encode("utf-8"))
                    for idx in self.listeners:
                        self.listeners[idx](line)
            except:
                break

    def feed(self, data):
        self.pout.write(data.encode("utf-8"))

    def dumps(self):
        return "\n".join(self.screen.display)

    def kill(self):
        os.killpg(self.pid, signal.SIGKILL)


def open_terminal(command="bash -i -l -s", columns=80, lines=24, cwd=None):
    if WINDOWS:
        return WindowsTerminal(command, columns, lines, cwd)
    else:
        return LinuxTerminal(command, columns, lines, cwd)
