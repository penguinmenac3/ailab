# AI Lab

AI Lab tries to make developing neural networks easier. It is written with the major frameworks keras, tensorflow 2 and pytorch 1 in mind.

Whilst the implementation of the model and training loop differ a lot between the frameworks, there is some common ground:
1. Loading an preparing the data
2. Experiment management (multiple configurations, training servers)
3. Visualization

Details on what is common here will be explained after a short installation instruction.

## Installation

1. Install [tensorflow 2](https://www.tensorflow.org/install) or [pytorch 1](https://pytorch.org/get-started/locally/) according to their instructions.

2. Simply pip install this package:

```bash
pip install ailab
```

## 1. Loading and preparing the data

At the core of each ai problem is some dataset.
The dataset exploration, loading and preparation is to a large degree framework independant.
Only the last step - loading it on the gpu - is framework dependant.
Therefor, a shared dataloading methodology, which is compatible with all frameworks is provided.

```python
from ailab.data import DataProvider

class MyDataset(DataProvider):
    def __init__(self, config, phase):
        super().__init__(config, phase)
    
    def __len__(self):
        return 42

    def __getitem__(self, index):
        features = {"foo":[1,2,3]}
        labels = {"bar":[3,2,1], "baz": [4,5,6]}
        return features, labels
```

This dataset can then be transformed, batched and converted to the framework. In this case we will use keras.

```python
from ailab.data import BatchedDataProvider
from ailab.data.keras import Sequence

dataset = MyDataset(config, phase)
batched_dataset = BatchedDataProvider(config, dataset)
keras_sequence = Sequence(config, batched_dataset)
assert isinstance(keras_sequence, tf.keras.utils.Sequence)
```

## 2. Experiment management

In real world scenarios your first approach typically does not work.
You will need a lot of iterations until it works.
During those iterations it will happen, that you make progress towards your goal and that you step backwards.
When you have a cluster and multiple gpus at hand, running multiple experiments in paralell can be a time saver.
However, keeping track of what exirement was executed with which code/configuration is difficult and prone to errors.

AI Lab provides easy experiment management.
Ranging from running multiple experiments on different machines and monitoring their progress via the visualization module, or keeping track of configurations.

Firstly, it is encouraged to have all your training configuration parameters in one central place and not scattered all across your code.
This will make extracting your training configuration for your paper easier.
Simply subcalss the Config class.
```python
from ailab.experiments import Config
class MyConfig(Config):
    def __init__(self):
        self.train.batch_size = 42
        #...
        self.arch.model = MyModel
        self.arch.loss = MyLoss
        self.arch.metrics = MyMetrics
        self.arch.prepare = MyDataset

        # avoid errors during training -> check completeness
        self.check_completeness()  # preimplemented
```

Keeping track of configurations can be done by two lines of code, which you will have to add befor your training loop.

```python
from ailab.experiment import backup
backup(os.path.join(checkpoint_dir, "src"))
```

There are also other handy functions to check if your code is in sync with a backup `needs_backup(path_to_backup)` and to load a backup into your pythonpath so you can import straight from your backup `load_backup(path_to_backup)`.

## 3. Visualization

Everything that is inside the checkpoint folder is availible to visualization.
You can directly write data in a compatible format - images can be automatically detected.

Easier is probably using the visualization api:

```python
from ailab.visualization import log_fig, log_dict, log_scalar

# use matplotlib
import matplotlib.pyplot as plt
plt.plot(x, y, ...)
log_fig("MyFigure")  # outputs MyFigure.png
# or for numpy arrays of shape (h,w,1) or (h,w,3)
log_fig("NumpyImage", my_image)  # outputs NumpyImage.png

# log scalars
log_dict("Losses", losses)  # appends to Losses.csv
log_dict("Metrics", metrics)  # appends to Metrics.csv
log_scalar("Learning Rate", lr)  # appends to Learning_Rate.csv
```
You can use custom scripts to use the visualization data or simply use the visualization server with it`s webgui.

#### Visualization server
AI Lab Visualization consists of a ui and a server.
Since the ui is a static website that works on your local webbrowser no installation is needed. The static website is hosted [here](http://ailab.f-online.net/).

Running is as simple as running the module in python providing a path to a config file.

```bash
python -m ailab.visualization my_config.json
```

A config file must contain a host or * for any interface, a port, a list of users as a map and a path to your checkpoints.
(Typically the checkpoint path is on a network share, where all computers add their checkpoints and this pc reads them.)

```json
{
  "host": "*",
  "port": 12345,
  "users": 
  {
    "admin": "CHANGE_THIS"
  },
  "checkpoints": "/data/$USER/checkpoints"
}
```

### Privacy

All connection data is stored locally in your webbrowser and nothing is transmitted to the host of ailab ui.
There is only direct communication between your webbroser and the server you add via the "Add Server" Dialog.

The servers you add are not controlled by us and therefore can do whatever they want with your data.
However, when the servers are owned/run by you and use the official ailab-server software, they will not track activities or report back information to a third party.

Even though this sounds pretty safe, there is yet no ssl implementation for the connection to your servers, keep that in mind.
(If you know how to implement an easy to use ssl on the client and the server, I will be happy to receive your pull request.)
