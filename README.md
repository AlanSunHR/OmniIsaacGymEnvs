# Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim

### Installation

Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html) to install the latest Isaac Sim release.

Once installed, this repository can be used as a python module, `omniisaacgymenvs`, with the python executable provided in Isaac Sim.

To install `omniisaacgymenvs`, first clone this repository:

```bash
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
```

Once cloned, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html). By default, this should be `python.sh`. We will refer to this path as `PYTHON_PATH`.

To set a `PYTHON_PATH` variable in the terminal that links to the python executable, we can run a command that resembles the following. Make sure to update the paths to your local path.

```
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
```

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:

```bash
PYTHON_PATH -m pip install -e .
```


### Running the testing task

```
PYTHON_PATH scripts/random_policy.py task=TestModule num_envs=1
```

### Process of the task

The task is used to test if the closed chain robot will be simulated normally in Omniverse Isaac Gym. The total episode length is 1000. At the first episode, the robot will be reset to its initial pose with 0 velocity for every DoF. For every 200 timesteps, a random DoF position target (directly from the actor network output, inside the range of [-1, 1]) will be set, and ideally the robot will move the joints to the target.


