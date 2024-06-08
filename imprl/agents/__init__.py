import imprl.agents

from imprl.agents.DDQN import DDQNAgent as DDQN
from imprl.agents.JAC import JointActorCritic as JAC
from imprl.agents.DCMAC import DeepCentralisedMultiAgentActorCritic as DCMAC
from imprl.agents.DDMAC import DeepDecentralisedMultiAgentActorCritic as DDMAC
from imprl.agents.IACC import IndependentActorCentralisedCritic as IACC
from imprl.agents.IAC import IndependentActorCritic as IAC
from imprl.agents.IAC_PS import IndependentActorCriticParameterSharing as IAC_PS
from imprl.agents.IACC_PS import (
    IndependentActorCentralisedCriticParameterSharing as IACC_PS,
)
from imprl.agents.VDN_PS import ValueDecompositionNetworkParameterSharing as VDN_PS
from imprl.agents.QMIX_PS import QMIXParameterSharing as QMIX_PS


def get_agent_class(algorithm):
    try:
        return getattr(imprl.agents, algorithm)
    except AttributeError:
        raise NotImplementedError(f"The algorithm '{algorithm}' is not implemented.")
