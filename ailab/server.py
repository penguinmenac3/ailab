import os
import traceback
import datetime
import signal
import subprocess
import json
from time import sleep, time, strftime, gmtime
from threading import Thread
from functools import partial
import GPUtil
import psutil
from ailab.terminal_emulator import open_terminal

TICKRATE = 0.1
PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", ".git"]


class Server(object):
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.server_load = {"cpu": 0, "ram": 0, "gpus": []}
        self.running = True
        self.current_task = None
        self.username = None
        self.terminals = {}
        self.processes = {}
        self.update_terminals = []
        self.ignore_list = PYTHON_IGNORE_LIST
        Thread(target=self.run).start()

    def run(self):
        while self.running:
            cpu = int(psutil.cpu_percent() * 10) / 10
            ram = int(psutil.virtual_memory().percent * 10) / 10
            gpus = [{"id": gpu.id, "GPU": int(gpu.load * 1000) / 10, "MEM": int(gpu.memoryUtil * 1000) / 10}
                    for gpu in GPUtil.getGPUs()]
            self.server_load = {"cpu": cpu, "ram": ram, "gpus": gpus}
            sleep(2)

    def setup(self, state, entanglement):
        state["server_load"] = None
        state["experiment"] = None
        state["term_height"] = 80
        state["term_width"] = 24
        self.username = entanglement.username
        print(self.username)
        entanglement.set_experiment = partial(self.set_experiment, state, entanglement)
        entanglement.add_experiment = partial(self.add_experiment, state, entanglement)
        entanglement.save_file = partial(self.save_file, state, entanglement)
        entanglement.open_file = partial(self.open_file, state, entanglement)
        entanglement.get_files = partial(self.get_files, state, entanglement)
        entanglement.run_file = partial(self.run_file, state, entanglement)
        entanglement.create_term = partial(self.create_term, state, entanglement)
        entanglement.send_terminal = partial(self.send_terminal, state, entanglement)
        entanglement.close_term = partial(self.close_term, state, entanglement)
        entanglement.resize_term = partial(self.resize_term, state, entanglement)
        for exp_name in self.config["projects"]:
            entanglement.remote_fun("update_experiment")(exp_name, "ready")

    def run_file(self, state, entanglement, file):
        working_dir = self.config["projects"][state["experiment"]]
        filename = os.path.join(working_dir, file)
        cmd = None
        if file.endswith(".py"):
            cmd = "python -m " + file
        else:
            cmd = filename
        if cmd is not None:
            self.create_term_with_cmd(state, entanglement, cmd)

    def create_term_with_cmd(self, state, entanglement, cmd):
        if cmd is None:
            cmd = "bash -i -l -s"
        project = state["experiment"]
        working_dir = self.config["projects"][state["experiment"]]
        if project not in self.terminals:
            self.terminals[project] = {}
        if project not in self.processes:
            self.processes[project] = {}
        term = open_terminal(command=cmd, cwd=working_dir, columns=state["term_width"], lines=state["term_height"])
        self.terminals[project][term] = ""
        self.processes[project][term.pid] = term

    def send_terminal(self, state, entanglement, process_id, inp):
        project = state["experiment"]
        if process_id in self.processes[project]:
            execProcess = self.processes[project][process_id]
            execProcess.feed(inp)
        else:
            print("Tried to write to non existent process.")

    def resize_term(self, state, entanglement, lines, columns):
        project = state["experiment"]
        state["term_height"] = lines
        state["term_width"] = columns
        if project in self.processes:
            for process_id in self.processes[project]:
                execProcess = self.processes[project][process_id]
                execProcess.resize(lines, columns)

    def create_term(self, state, entanglement):
        self.create_term_with_cmd(state, entanglement, None)

    def close_term(self, state, entanglement, pid):
        project = state["experiment"]
        execProcess = None
        proc_idx = 0
        for idx in self.processes[project]:
            execProcess = self.processes[project][idx]
            if execProcess.pid == pid:
                proc_idx = idx
                break
        if execProcess is not None:
            print("Killing process {}".format(execProcess.pid))
            execProcess.kill()
            if proc_idx in self.processes[project]:
                del self.processes[project][proc_idx]
            else:
                print("PID does not exist in processes {}".format(proc_idx))
            if execProcess in self.terminals[project]:
                del self.terminals[project][execProcess]
            else:
                print("execProcess does not exist in terminals")
            # print("process killed {}".format(execProcess.returncode))
            entanglement.remote_fun("update_terminal")(project, execProcess.pid, None)

    def set_experiment(self, state, entanglement, name):
        entanglement.remote_fun("reset_experiment")()
        entanglement.experiment_title = name
        state["experiment"] = name
        t = strftime("%Y-%m-%d %H:%M:%S", gmtime(time()))
        entanglement.remote_fun("update_experiment_events")([{"time": t, "event": "Test"}])
        if name in self.terminals:
            for execProcess in self.terminals[name]:
                entanglement.remote_fun("update_terminal")(name, execProcess.pid, self.terminals[name][execProcess])

    def add_experiment(self, state, entanglement, name, path):
        path = os.path.join(self.config["workspace"], path).replace("\\", "/")
        os.makedirs(path, exist_ok=True)
        self.config["projects"][name] = path
        config_str = json.dumps(self.config, indent=4, sort_keys=True)
        with open(self.config_path, "w") as f:
            f.write(config_str)
        entanglement.remote_fun("update_experiment")(name, "ready")

    def __ignore(self, candidate, forbidden_list):
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

    def get_files(self, state, entanglement):
        filelist = []
        if state["experiment"] is None:
            return
        for path, subdirs, files in os.walk(self.config["projects"][state["experiment"]]):
            files = [x for x in files if not self.__ignore(x, self.ignore_list)]
            subdirs[:] = [x for x in subdirs if not self.__ignore(x, self.ignore_list)]
            for name in files:
                file = os.path.join(path, name)
                file = file.replace("\\", "/")
                file = file.replace(self.config["projects"][state["experiment"]], ".")
                filelist.append(file)
        entanglement.remote_fun("update_files")(filelist)

    def save_file(self, state, entanglement, name, content):
        with open(os.path.join(self.config["projects"][state["experiment"]], name), "w") as f:
            f.write(content)

        self.lint(state, entanglement, name)

    def open_file(self, state, entanglement, name):
        filetype = "python"
        if name.endswith(".sh"):
            filetype = "shell"
        if name.endswith(".c"):
            filetype = "c"
        if name.endswith(".h"):
            filetype = "c"
        if name.endswith(".cpp"):
            filetype = "cpp"
        if name.endswith(".hpp"):
            filetype = "cpp"
        if name.endswith(".js"):
            filetype = "javascript"
        if name.endswith(".json"):
            filetype = "javascript"
        content = ""
        try:
            with open(os.path.join(self.config["projects"][state["experiment"]], name), "r") as f:
                content = f.read()
        except:
            content = "Error reading file"
        entanglement.remote_fun("update_file")({"name": name, "content": content, "type": filetype})
        self.lint(state, entanglement, name)

    def lint(self, state, entanglement, name):
        file = os.path.join(self.config["projects"][state["experiment"]], name)
        linter_result = "TODO linter not implemented."
        entanglement.remote_fun("update_linter")({"name": name, "content": linter_result})

    def tick(self, state, entanglement):
        if self.server_load != state["server_load"]:
            state["server_load"] = self.server_load
            entanglement.remote_fun("update_server_status")(self.server_load)

        project = state["experiment"]
        if project is not None and project in self.terminals:
            for execProcess in self.terminals[project]:
                text = execProcess.dumps()
                if "tmp" not in state:
                    state["tmp"] = {}
                if project not in state["tmp"]:
                    state["tmp"][project] = {}
                if execProcess not in state["tmp"][project]:
                    state["tmp"][project][execProcess] = ""
                if state["tmp"][project][execProcess] != text:
                    state["tmp"][project][execProcess] = text
                    entanglement.remote_fun("update_terminal")(project, execProcess.pid, text)

    def on_entangle(self, entanglement):
        state = {}
        # Read experiments automatically?
        if "auto_projects" in self.config and self.config["auto_projects"]:
            folders = [os.path.join(self.config["workspace"], f) for f in os.listdir(self.config["workspace"])]
            folders = [f for f in folders if os.path.isdir(f)]
            names = [f.split(os.sep)[-1].replace("-", " ").replace("_", " ") for f in folders]
            self.config["projects"] = dict(zip(names, folders))
            print("Automatically detected projects: {}".format(self.config["projects"]))

        self.setup(state, entanglement)
        try:
            while True:
                self.tick(state, entanglement)
                sleep(TICKRATE)
        except:
            with open("exceptions.log", "a") as log:
                log.write("%s: Exception occurred:\n" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                traceback.print_exc(file=log)

        entanglement.close()
