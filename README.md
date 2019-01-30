# AI Lab Server

The server backend for AI Lab.
AI Lab consists of a ui and a server.
Since the ui is a static website that works on your local webbrowser no installation is needed. Just open [TODO](#)

## Capabilities

AI Lab is a tool that let's you manage multiple projects on local and remote machines.
You can see experiments and their status as well as the server states on the front page.

Once you selected a project, you can browse the files and edit them.
You can see recent events of this project.
And of course you can run python and shell files in a terminal as well as just use a terminal on the machine starting in the project folder.

## Install Server

Installation can be done via pip.

```bash
pip install ailab
```

## Run Server

Running is as simple as running the module in python providing a path to a config file.

```bash
python -m ailab my_config.json
```

A config file must contain a host or * for any interface, a port, a list of users as a map and a list of projects as a map of name to folder.

```json
{
  "host": "*",
  "port": 12345,
  "users": 
  {
    "admin": "CHANGE_THIS"
  },
  "projects": {
      "MyCoolWindowsProject": "D:/myfolder",
      "MyCoolLinuxProject": "/home/myuser/myfolder"
  }
}
```

## Privacy

All connection data is stored locally in your webbrowser and nothing is transmitted to the host of ailab ui.
There is only direct communication between your webbroser and the server you add via the "Add Server" Dialog.

The servers you add are not controlled by us and therefore can do whatever they want with your data.
However, when the servers are owned/run by you and use the official ailab-server software, they will not track activities or report back information to a third party.

Even though this sounds pretty safe, there is yet no ssl implementation for the connection to your servers, keep that in mind.
(If you know how to implement an easy to use ssl on the client and the server, I will be happy to receive your pull request.)
