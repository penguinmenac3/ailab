import os
import traceback
import datetime
from typing import Dict, Any, List, Sequence
import signal
import subprocess
import json
import base64
from time import sleep, time, strftime, gmtime
from threading import Thread
from functools import partial
import GPUtil
import psutil
from entangle.entanglement import Entanglement
from ailab.server.terminal_emulator import open_terminal
from rempy.server import Server as RempyServer

TICKRATE = 0.1
PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", ".git"]
DEFAULT_RESULT_EVENT = {"timestamp": None, "goal": None, "progress": None, "eta": None, "score": None}

class Server(object):
    def __init__(self, config: Dict[str, Any], config_path: str) -> None:
        """
        Creates an ailab server.

        :param config: The configuration for the server as a dictionary.
        :param config_path: The path where the config file is stored, so it can be overwritten on change.
        """
        self.config = config
        self.config_path = config_path
        self.server_load = {"cpu": 0, "ram": 0, "gpus": []}
        self.projects = {}
        self.results = {}
        self.latest_result_events = {}
        self.running = True
        self.current_task = None
        self.username = None
        self.terminals = {}
        self.processes = {}
        self.update_terminals = []
        self.ignore_list = PYTHON_IGNORE_LIST
        self.rempy_server = RempyServer(config)
        Thread(target=self.run).start()

    def run(self) -> None:
        """
        Run the server.
        """
        while self.running:
            cpu = int(psutil.cpu_percent() * 10) / 10
            ram = int(psutil.virtual_memory().percent * 10) / 10
            gpus = [{"id": gpu.id, "GPU": int(gpu.load * 1000) / 10, "MEM": int(gpu.memoryUtil * 1000) / 10}
                    for gpu in GPUtil.getGPUs()]
            self.server_load = {"cpu": cpu, "ram": ram, "gpus": gpus}

            # Read experiment log and get latest event.
            for result_name in self.results:
                log_fname = os.path.join(self.results[result_name], "ailab.log")
                if not os.path.exists(log_fname):
                    log_fname = os.path.join(self.results[result_name], "log.txt")
                if os.path.exists(log_fname):
                    with open(log_fname, "r") as f:
                        lines = f.readlines()
                    event = lines[-1]
                    event = json.loads(event)
                    self.latest_result_events[result_name] = event
                elif result_name in self.latest_result_events:
                    del self.latest_result_events[result_name]
            sleep(2)

    def setup(self, state: Dict[str, Any], entanglement: Entanglement) -> None:
        """
        Setup the server. This function registers all the local functions to the entanglement.
        :param state: The state object which is shared.
        :param entanglement: The entanglement for the connection that should be setup.
        """
        state["server_load"] = None
        state["experiment"] = None
        state["term_height"] = 80
        state["term_width"] = 24
        state["latest_result_events"] = {}
        self.username = entanglement.username
        print(self.username)
        entanglement.set_experiment = partial(self.set_experiment, state, entanglement)
        entanglement.new_project = partial(self.new_project, state, entanglement)
        entanglement.save_file = partial(self.save_file, state, entanglement)
        entanglement.open_file = partial(self.open_file, state, entanglement)
        entanglement.get_files = partial(self.get_files, state, entanglement)
        entanglement.run_file = partial(self.run_file, state, entanglement)
        entanglement.create_term = partial(self.create_term, state, entanglement)
        entanglement.send_terminal = partial(self.send_terminal, state, entanglement)
        entanglement.close_term = partial(self.close_term, state, entanglement)
        entanglement.resize_term = partial(self.resize_term, state, entanglement)
        for exp_name in self.projects:
            entanglement.remote_fun("update_experiment")(exp_name, "project")
        for result_name in self.results:
            if result_name not in self.latest_result_events:
                self.latest_result_events[result_name] = DEFAULT_RESULT_EVENT
            if result_name not in state["latest_result_events"] or self.latest_result_events[result_name] != state["latest_result_events"][result_name]:
                state["latest_result_events"][result_name] = self.latest_result_events[result_name]
                entanglement.remote_fun("update_experiment")(result_name, state["latest_result_events"][result_name])

    def run_file(self, state: Dict[str, Any], entanglement: Entanglement, file: str) -> None:
        working_dir = self.experiments[state["experiment"]]
        filename = os.path.join(working_dir, file)
        cmd = None
        if file.endswith(".py"):
            cmd = "python -m " + file
        else:
            cmd = filename
        if cmd is not None:
            self.create_term_with_cmd(state, entanglement, cmd)

    def create_term_with_cmd(self, state: Dict[str, Any], entanglement: Entanglement, cmd) -> None:
        if cmd is None:
            cmd = "bash -i -l -s"
        project = state["experiment"]
        working_dir = self.experiments[state["experiment"]]
        if project not in self.terminals:
            self.terminals[project] = {}
        if project not in self.processes:
            self.processes[project] = {}
        term = open_terminal(command=cmd, cwd=working_dir, columns=state["term_width"], lines=state["term_height"])
        self.terminals[project][term] = ""
        self.processes[project][term.pid] = term

    def send_terminal(self, state: Dict[str, Any], entanglement: Entanglement, process_id: str, inp: str) -> None:
        project = state["experiment"]
        if process_id in self.processes[project]:
            execProcess = self.processes[project][process_id]
            execProcess.feed(inp)
        else:
            print("Tried to write to non existent process.")

    def resize_term(self, state: Dict[str, Any], entanglement: Entanglement, lines: int, columns: int) -> None:
        project = state["experiment"]
        state["term_height"] = lines
        state["term_width"] = columns
        if project in self.processes:
            for process_id in self.processes[project]:
                execProcess = self.processes[project][process_id]
                execProcess.resize(lines, columns)

    def create_term(self, state: Dict[str, Any], entanglement: Entanglement) -> None:
        self.create_term_with_cmd(state, entanglement, None)

    def close_term(self, state: Dict[str, Any], entanglement: Entanglement, pid: str) -> None:
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

    def set_experiment(self, state: Dict[str, Any], entanglement: Entanglement, name: str):
        entanglement.remote_fun("reset_experiment")()
        entanglement.experiment_title = name
        state["experiment"] = name
        if name in self.terminals:
            for execProcess in self.terminals[name]:
                entanglement.remote_fun("update_terminal")(name, execProcess.pid, self.terminals[name][execProcess])

    def new_project(self, state: Dict[str, Any], entanglement: Entanglement, name: str, path: str):
        path = os.path.join(self.config["workspace"], path).replace("\\", "/")
        os.makedirs(path, exist_ok=True)
        self.experiments[name] = path
        self.projects[name] = path
        config_str = json.dumps(self.config, indent=4, sort_keys=True)
        with open(self.config_path, "w") as f:
            f.write(config_str)
        entanglement.remote_fun("update_experiment")(name, "project")

    def __ignore(self, candidate: str, forbidden_list: List[str]) -> bool:
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

    def get_files(self, state: Dict[str, Any], entanglement: Entanglement) -> None:
        filelist = []
        if state["experiment"] is None:
            return
        for path, subdirs, files in os.walk(self.experiments[state["experiment"]]):
            files = [x for x in files if not self.__ignore(x, self.ignore_list)]
            subdirs[:] = [x for x in subdirs if not self.__ignore(x, self.ignore_list)]
            for name in files:
                file = os.path.join(path, name)
                file = file.replace(self.experiments[state["experiment"]], ".")
                file = file.replace("\\", "/")
                # In case user used unix format on windows do it after conversion.
                file = file.replace(self.experiments[state["experiment"]], ".")
                filelist.append(file)
        entanglement.remote_fun("update_files")(filelist)

    def save_file(self, state: Dict[str, Any], entanglement: Entanglement, name: str, content: str) -> None:
        if name.endswith(".png"):
            return
        with open(os.path.join(self.experiments[state["experiment"]], name), "w") as f:
            f.write(content)

        self.lint(state, entanglement, name)

    def open_file(self, state: Dict[str, Any], entanglement: Entanglement, name: str) -> None:
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
        if name.endswith(".png"):
            filetype = "image"
        content = ""
        if filetype == "image":
            try:
                with open(os.path.join(self.experiments[state["experiment"]], name), "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")
                    print(content)
            except:
                content = ""
        else:
            try:
                with open(os.path.join(self.experiments[state["experiment"]], name), "r") as f:
                    content = f.read()
            except:
                content = "Error reading file"
        entanglement.remote_fun("update_file")({"name": name, "content": content, "type": filetype})
        if filetype != "image":
            self.lint(state, entanglement, name)

    def lint(self, state: Dict[str, Any], entanglement: Entanglement, name: str) -> None:
        file = os.path.join(self.experiments[state["experiment"]], name)
        linter_result = "TODO linter not implemented."
        entanglement.remote_fun("update_linter")({"name": name, "content": linter_result})

    def tick(self, state: Dict[str, Any], entanglement: Entanglement) -> None:
        if self.server_load != state["server_load"]:
            state["server_load"] = self.server_load
            entanglement.remote_fun("update_server_status")(self.server_load)

        for result_name in self.latest_result_events:
            if result_name not in state["latest_result_events"] or self.latest_result_events[result_name] != state["latest_result_events"][result_name]:
                state["latest_result_events"][result_name] = self.latest_result_events[result_name]
                entanglement.remote_fun("update_experiment")(result_name, state["latest_result_events"][result_name])

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

    def find_results(self, folder: str, max_depth: int, depth: int=0) -> List [str]:
        folders = []
        if depth < max_depth:
            candidate_folders = [os.path.join(folder, f) for f in os.listdir(folder)]
            candidate_folders = [f for f in candidate_folders if os.path.isdir(f)]
            for candidate in candidate_folders:
                log_fname = os.path.join(candidate, "ailab.log")
                if not os.path.exists(log_fname):
                    log_fname = os.path.join(candidate, "log.txt")
                if os.path.exists(log_fname):
                    folders.append(candidate)
                else:
                    folders.extend(self.find_results(candidate, max_depth, depth=depth+1))
        return folders

    def on_entangle(self, entanglement: Entanglement) -> None:
        state = {}
        protocol = entanglement.get("protocol")
        if protocol == "rempy":
            self.rempy_server.callback(entanglement)
        else:
            # Read experiments automatically?
            if "auto_detect_projects" in self.config and self.config["auto_detect_projects"]:
                folders = [os.path.join(self.config["workspace"], f) for f in os.listdir(self.config["workspace"])]
                folders = [f for f in folders if os.path.isdir(f)]
                names = [f.split(os.sep)[-1].replace("-", " ").replace("_", " ") for f in folders]
                self.projects = dict(zip(names, folders))
            else:
                self.projects = self.config["projects"]
                
            folders = self.find_results(self.config["results"], max_depth=5)
            names = []
            for name in folders:
                name = name.replace(self.config["results"], "")
                name = name.replace("-", " ").replace("_", " ")
                if name.startswith("/"):
                    name = name[1:]
                names.append(name)
            self.results = dict(zip(names, folders))
            self.experiments = {**self.projects, **self.results}

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
