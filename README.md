# AI Lab

AI Lab tries to make developing neural networks easier. It is written with the major frameworks keras, tensorflow 2 and pytorch 1 in mind.

Whilst the implementation of the model and training loop differ a lot between the frameworks, there is some common ground:
1. Loading an preparing the data
2. Experiment management (multiple configurations, training servers)
3. Visualization

Details on what is common here will be explained after a short installation instruction.

## Installation

First install [keras](), [tensorflow]() or [pytorch]() according to their instructions.
After that simply pip install this package.

```bash
pip install ailab
```

## 1. Loading and preparing the data


## 2. Experiment management


## 3. Visualization

AI Lab Visualization consists of a ui and a server.
Since the ui is a static website that works on your local webbrowser no installation is needed. The static website is hosted [here](http://ailab.f-online.net/).

Running is as simple as running the module in python providing a path to a config file.

```bash
python -m ailab.visualization my_config.json
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
  "workspace": "/home/myuser/workspace",
  "projects": {
      "MyCoolWindowsProject": "D:/myfolder",
      "MyCoolLinuxProject": "/home/myuser/myfolder"
  }
}
```

### Privacy

All connection data is stored locally in your webbrowser and nothing is transmitted to the host of ailab ui.
There is only direct communication between your webbroser and the server you add via the "Add Server" Dialog.

The servers you add are not controlled by us and therefore can do whatever they want with your data.
However, when the servers are owned/run by you and use the official ailab-server software, they will not track activities or report back information to a third party.

Even though this sounds pretty safe, there is yet no ssl implementation for the connection to your servers, keep that in mind.
(If you know how to implement an easy to use ssl on the client and the server, I will be happy to receive your pull request.)
