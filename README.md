# AI Lab

AI Lab is a gui and a server backend.
The GUI provides an overview over experiments and their status, based on logfiles on the disk.
The server provides functionalities for the gui such as parsing the logfiles, getting the usage of the system and providing the files to view in the gui.
Furthermore, the server also provides a rempy host, where jobs can be submitted for exection.

## Installation

1. Simply pip install this package from git:

```bash
pip install ailab
```

## Running: AI Lab Server

AI Lab Visualization consists of a ui and a server.
Since the ui is a static website that works on your local webbrowser no installation is needed. The static website is hosted [here](http://ailab.f-online.net/).

Running is as simple as running the module in python providing a path to a config file.

```bash
ailab my_config.json
```

A config file must contain a host or * for any interface, a port, a list of users as a map and a path to your checkpoints.
(Typically the checkpoint path is on a network share, where all computers add their checkpoints and this pc reads them.)
The list of `gpus` gives you the opportunity to limit the gpus ailab will assign for scheduled tasks.
The gpu ids are equivalent to the numbers used for `CUDA_VISIBLE_DEVICES`.

```json
{
  "host": "*",
  "port": 12345,
  "users": 
  {
    "admin": "CHANGE_THIS"
  },
  "workspace": "/home/$USER/Git",
  "results": "/home/$USER/Results",
  "queue": "/tmp/ailab_queue",
  "auto_detect_experiments": false,
  "projects": {

  },
  "gpus": [0]
}
```

### Privacy

All connection data is stored locally in your webbrowser and nothing is transmitted to the host of ailab ui.
There is only direct communication between your webbroser and the server you add via the "Add Server" Dialog.

The servers you add are not controlled by us and therefore can do whatever they want with your data.
However, when the servers are owned/run by you and use the official ailab-server software, they will not track activities or report back information to a third party.

Even though this sounds pretty safe, there is yet no ssl implementation for the connection to your servers, keep that in mind.
(If you know how to implement an easy to use ssl on the client and the server, I will be happy to receive your pull request.)
