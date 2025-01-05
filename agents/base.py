import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, state_dim, action_dim, discrete=True):
        """Initialize the agent.
        
        Args:
            state_dim: Dimension of state space (int or tuple)
            action_dim: Dimension of action space
            discrete: Whether action space is discrete
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        
        # Initialize networks
        self.build_networks()
        
    @abstractmethod
    def build_networks(self):
        """Build neural networks required for agent."""
        pass
        
    @abstractmethod
    def get_action(self, state, training=True):
        """Return an action given a state."""
        pass
    
    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Perform a single training step."""
        pass
    
    def save(self, filepath):
        """Save agent models to disk."""
        pass
    
    def load(self, filepath):
        """Load agent models from disk."""
        pass

class DiscreteActionAgent(BaseAgent):
    """Base class for agents with discrete action spaces."""
    
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, discrete=True)
        
    def epsilon_greedy_policy(self, state, epsilon=0.0):
        """Epsilon greedy policy for action selection.
        
        Args:
            state: Current state
            epsilon: Probability of random action
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.get_q_values(state)
            return int(tf.argmax(q_values[0]))
    
    @abstractmethod
    def get_q_values(self, state):
        """Get Q-values for all actions in current state."""
        pass
