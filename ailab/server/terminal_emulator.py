import json
import os
import shlex
import signal
import webbrowser
from pathlib import Path
from threading import Thread
from typing import Callable
WINDOWS = False
try:
    import pyte
    import pty
except:
    WINDOWS = True


class Terminal(object):
    def add_listener(self, listener: Callable[[str], None]) -> None:
        raise NotImplementedError()

    def remove_listener(self, listener: Callable[[str], None]) -> None:
        raise NotImplementedError()

    def resize(self, lines: int, columns: int) -> None:
        raise NotImplementedError()

    def feed(self, data: str) -> None:
        raise NotImplementedError()

    def dumps(self) -> str:
        raise NotImplementedError()

    def kill(self) -> None:
        raise NotImplementedError()


class WindowsTerminal(Terminal):
    def __init__(self, command: str, columns: int, lines: int, cwd: str = None):
        pass


class LinuxTerminal(Terminal):
    def __init__(self, command: str, columns: int, lines: int, cwd: str = None) -> None:
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

    def add_listener(self, listener: Callable[[str], None]) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[str], None]) -> None:
        self.listeners.remove(listener)

    def resize(self, lines: int, columns: int) -> None:
        self.screen.resize(lines, columns)

    def read_proc(self) -> None:
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

    def feed(self, data: str) -> None:
        self.pout.write(data.encode("utf-8"))

    def dumps(self) -> str:
        return "\n".join(self.screen.display)

    def kill(self) -> None:
        os.killpg(self.pid, signal.SIGKILL)


def open_terminal(command: str = "bash -i -l -s", columns: int = 80, lines: int = 24, cwd: str = None) -> Terminal:
    if WINDOWS:
        return WindowsTerminal(command, columns, lines, cwd)
    else:
        return LinuxTerminal(command, columns, lines, cwd)
