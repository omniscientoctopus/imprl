# Inspection and Maintenance Planning using Reinforcement Learning (IMPRL)

A (beginner-friendly) library for applying reinforcement learning to inspection and maintenance planning of deteriorating engineering systems. This library was primarily developed as a pedogogic excercise and for research use.

## Installation

### 1. Create a virtual environment

```bash
conda create --name imprl_env -y python==3.9
conda activate imprl_env
```

### 2. Install the dependencies

```bash
pip install poetry==1.8
poetry install
```

Following best practices, `poetry install` installs the dependencies from the `poetry.lock` file. This file rigorously specifies all the dependencies required to build the library. It ensures that the project does not break because of unexpected changes in (transitive) dependencies ([more info](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)).

<details>
<summary>Installing additional packages</summary>

You can them add via `poetry add` ([official docs](https://python-poetry.org/docs/cli/#add)) in the command line. 

For example, to install [Jupyter notebook](https://pypi.org/project/notebook/),

```bash 
# Allow >=7.1.2, <8.0.0 versions
poetry add notebook@^7.1.2
```
This will resolve the package dependencies (and adjust versions of transitive dependencies if necessary) and install the package. If the package dependency cannot be resolved, try to relax the package version and try again.
</details>

### 3. Setup wandb

```bash
wandb login
# <enter wandb API key>
```


## RL Agents

The following (multiagent) reinforcement algorithms are implemented,
  - Double Deep Q-Network (DDQN)
  - Joint Actor Critic (JAC)
  - Deep Centralized Multiagent Actor Critic (DCMAC)
  - Deep Decentralized Multiagent Actor Critic (DDMAC)
  - Independent Actor Centralized Critic (IACC)
  - Independent Actor Centralized Critic with Paramater Sharing (IACC-PS)
  - Independent Actor Critic (IAC)
  - Independent Actor Critic with Paramater Sharing (IAC-PS)


## Acknowledgements

This project utilizes the clever abstractions in [EPyMARL](https://github.com/uoe-agents/epymarl) and the author would like to acknowledge the insights shared in [Reinforcement Learning Implementation Tips and Tricks](https://agents.inf.ed.ac.uk/blog/reinforcement-learning-implementation-tricks/) for developing this library.


## Related Work


- [IMP-MARL](https://github.com/moratodpg/imp_marl): a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.
    - What's different: 
        - Environments: (Correlated and uncorrelated) k-out-of-n systems and offshore wind structural systems.
        - RL solvers: Provides wrappers for interfacing with several (MA)RL libraries such as [EPyMARL](https://github.com/uoe-agents/epymarl), [Rllib](imp_marl/imp_wrappers/examples/rllib/rllib_example.py), [MARLlib](imp_marl/imp_wrappers/marllib/marllib_wrap_ma_struct.py) etc.
