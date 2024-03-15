# Inspection and Maintenance Planning using Reinforcement Learning (IMPRL)

A (beginner-friendly) library for applying reinforcement learning to inspection and maintenance planning of deteriorating engineering systems. This library was primarily developed as a pedogogic excercise and use for my research.

## Installation


## RL Agents

The following multiagent reinforcement algorithms are implemented,
  - Double Deep Q-Network (DDQN)
  - Joint Actor Critic (JAC)
  - Deep Centralized Multiagent Actor Critic (DCMAC)
  - Deep Decentralized Multiagent Actor Critic (DDMAC)
  - Independent Actor Centralized Critic (IACC)
  - Independent Actor Centralized Critic with Paramater Sharing (IACC-PS)
  - Independent Actor Critic (IAC)
  - Independent Actor Critic with Paramater Sharing (IAC-PS)


## Acknowledgements

This project utilizes the clever abstractions used in [EPyMARL](https://github.com/uoe-agents/epymarl) and the author would like to acknowledge the insights shared in [Reinforcement Learning Implementation Tips and Tricks](https://agents.inf.ed.ac.uk/blog/reinforcement-learning-implementation-tricks/) for developing this library.


## Related Work


- [IMP-MARL](https://github.com/moratodpg/imp_marl): a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.
    - What's different: 
        - Environments: (Correlated and uncorrelated) k-out-of-n systems and offshore wind structural systems.
        - RL solvers: Provides wrappers for interfacing with several (MA)RL libraries such as [EPyMARL](https://github.com/uoe-agents/epymarl), [Rllib](imp_marl/imp_wrappers/examples/rllib/rllib_example.py), [MARLlib](imp_marl/imp_wrappers/marllib/marllib_wrap_ma_struct.py) etc.

