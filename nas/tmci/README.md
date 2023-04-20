What is TMCI?
=============

TMCI is a generic interface for checkpointing and reloading Tensorflow models.
It is based on plugins that are loaded from dynamic libraries. TMCI allows
direct access to tensors memory to prevent data copies during checkpoints.

Installing TMCI on a workstation
--------------------------------

TMCI requires the following dependencies.
*  Python 3.6 or greater
*  Tensorflow 2.0 or greater
*  GCC 6.3.0 or a C++ compiler allowing the C++14 standard

Additionally, building custom plugins can be done using `cmake` (version 3.12 or
greater).

Start by cloning this repository.

```
git clone https://xgitlab.cels.anl.gov/sds/tmci.git
cd tmci
```

You will find a `tensorflow.json` file that can be edited to provide some
information about your tensorflow installation. If an entry of this file is set
to `null`, the TMCI setup script will try to guess it by importing tensorflow
and finding the location of its libraries.

Note: the extra CXXFLAGS that is there by default (`-D_GLIBCXX_USE_CXX11_ABI=0`)
is necessary if you have installed tensorflow using `pip`. Indeed the pip
package for tensorflow has been compiled with an old version of gcc (4.8) whose
ABI differs from newer versions. If you have compiled tensorflow yourself, you
should remove this flag.

Once you have edited tensorflow.json as needed, setup and install TMCI.

```
python setup.py install
```

Check that the installation went fine by opening a Python interactive session
and typing `import tmci`, which should return with no error.

Understanding TMCI and plugins
------------------------------

The best way to understand TMCI is to look at
[example/lenet5.py](https://xgitlab.cels.anl.gov/sds/tmci/blob/master/example/lenet5.py).
This python program creates a LeNet5 neural network, trains it, and evaluates it.
During training, it uses a `tmci.checkpoint.CheckpointCallback` object to
checkpoint the model at every epoch. After training, it uses the
`tmci.checkpoint.save_weights` function to save the model. It also reloads the
model using the `tmci.checkpoint.load_weights`. These are the only three
functionalities that TMCI offers.

TMCI itself does not implement checkpointing. The actual implementation of
checkpointing features is done through plugins. A plugin is a dynamic library
containing the definition of a class inheriting from
[tmci::Backend](https://xgitlab.cels.anl.gov/sds/tmci/blob/master/tmci/src/backend.hpp).
Such a backend implementation is identified by a *name* and must provide two
functions, `Save` and `Load`, which respectively store and reload the data
associated with a set of tensors. The constructor of such a backend implementation
must take a `const char*` null-terminated string representing plugin parameters
(it is up to the implementation to choose a particular format, such as
a serialized JSON dictionary, or a comma-separated list of parameters, etc.).

In [example/lenet5.py](https://xgitlab.cels.anl.gov/sds/tmci/blob/master/example/lenet5.py),
the `tmci.plugins.load` function loads the dynamic library containing the implementation
of a particular plugin (here *libdummy.so*). The
`tmci.checkpoint.CheckpointCallback` class, `tmci.checkpoint.save_weights`
function, and `tmci.checkpoint.load_weights` function then take a *backend*
string argument that identifies which plugin should be used for checkpointing or
reloading the data.

An example of such a plugin, called *dummy*, is provided in the
[plugin](https://xgitlab.cels.anl.gov/sds/tmci/tree/master/plugin) folder.
This folder is independent from the rest of the TMCI code, and is not built by
TMCI, so feel free to make a copy of it and modify its content when you want to
implement your own plugin.

The plugin folder is a cmake-based project. You can compile it as follows.
```
cd plugin
mkdir build
cd build
cmake ..
make
```
You should end up with a *libdummy.so* file in *plugin/build/src*.
You may need to edit line 16 of the CMakeLists.txt to replace
`TENSORFLOW_INCLUDE_DIR` with the location of the tensorflow include
directory in your system.

Let's now take a look at the source of the plugin.
[DummyBackend.hpp](https://xgitlab.cels.anl.gov/sds/tmci/blob/master/plugin/src/DummyBackend.hpp)
contains the definition of the *DummyBackend* class, which provides simple
Save and Load methods taking a vector of references to tensors, and print out
the address and the size (in bytes) of these tensors.
[DummyBackend.cpp](https://xgitlab.cels.anl.gov/sds/tmci/blob/master/plugin/src/DummyBackend.cpp)
calls the `TMCI_REGISTER_BACKEND` macro to register the *DummyBackend* class
as a backend for TMCI. This macro takes the name of the backend (without quotation
marks) as first argument, and the name of the class as second argument. The
name of the backend should follows the same naming rules as C variables (for
instance it should not contain spaces).

Once the dummy plugin is compiled, you may run the LeNet5 example from the
root directory of TMCI, as follows.
```
python example/lenet5.py
```
If everything goes well, you will see the network training for one epoch,
then the plugin will be invoked twice to checkpoint (first through the
`CheckpointCallback` class, then through the `save_weights` function) and once
to reload (through the `load_weights` function), displaying a list of 25 tensors
each time.

Installing and running TMCI on Theta
------------------------------------

Load the required modules.
```
module swap PrgEnv-intel PrgEnv-gnu
module load cce
module load datascience/tensorflow-2.0
module load cmake/3.14.5
```

Clone TMCI.
```
git clone https://xgitlab.cels.anl.gov/sds/tmci.git
cd tmci

```
Copy *theta/tensorflow.json* in the root of the source tree.
```
cp theta/tensorflow.json .
```

Build and install TMCI (locally).
```
python setup install --user
```

TMCI will be located in `~/.local/`.

Go to the plugin directory, copy the *CMakeLists.txt* file that is in the
*theta* directory, then make a *build* directory and build the plugin.
```
cd plugin
cp ../theta/CMakeLists.txt .
mkdir build
cd build
cmake ..
make
```

You should obtain a *libdummy.so* in *build/src*.

Go back to the root of the source tree and launch a job using qsub.
```
cd ../..
qsub -A <your-project> theta/run.qsub
```

*run.qsub* will first call *theta/common.sh* to load the required modules,
then call *example/lenet5.py* using `aprun`.