import tensorflow as tf
import numpy as np
from collections import deque
from .base import DiscreteActionAgent

class DQNAgent(DiscreteActionAgent):
    """Deep Q-Network agent."""
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=10000,
                 memory_size=10000,
                 batch_size=32,
                 target_update_freq=100,
                 double_dqn=True,
                 dueling=True):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay_steps: Number of steps to decay epsilon
            memory_size: Size of replay memory
            batch_size: Size of training batch
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use Double DQN
            dueling: Whether to use Dueling DQN
        """
        super().__init__(state_dim, action_dim)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling = dueling
        
        # Training state
        self.memory = deque(maxlen=memory_size)
        self.training_steps = 0
        
        # Build networks
        self.build_networks()
        
        # Initialize target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    def build_networks(self):
        """Build main and target Q-networks."""
        self.q_network = self._build_network()
        self.target_network = self._build_network()
    
    def _build_network(self):
        """Build a dueling or regular Q-network."""
        if not self.dueling:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=self.state_dim),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.action_dim)
            ])
        else:
            # Dueling Network
            inputs = tf.keras.layers.Input(shape=self.state_dim)
            hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
            hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
            
            # Value stream
            value = tf.keras.layers.Dense(32, activation='relu')(hidden)
            value = tf.keras.layers.Dense(1)(value)
            
            # Advantage stream
            advantage = tf.keras.layers.Dense(32, activation='relu')(hidden)
            advantage = tf.keras.layers.Dense(self.action_dim)(advantage)
            
            # Combine streams
            q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            return tf.keras.Model(inputs=inputs, outputs=q_values)
    
    def get_q_values(self, state):
        """Get Q-values for all actions in current state."""
        return self.q_network(state)
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy."""
        epsilon = self._get_epsilon() if training else 0.01
        return self.epsilon_greedy_policy(state, epsilon)
    
    def _get_epsilon(self):
        """Get current epsilon value based on training steps."""
        if self.training_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
               (self.training_steps / self.epsilon_decay_steps)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform a single training step using experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample batch
        batch = self._sample_batch()
        states, actions, rewards, next_states, dones = batch
        
        # Compute target Q-values
        if self.double_dqn:
            next_actions = tf.argmax(self.q_network(next_states), axis=1)
            next_q_values = tf.gather(
                self.target_network(next_states),
                next_actions,
                batch_dims=1
            )
        else:
            next_q_values = tf.reduce_max(self.target_network(next_states), axis=1)
            
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Update Q-network
        with tf.GradientTape() as tape:
            q_values = tf.gather(
                self.q_network(states),
                actions,
                batch_dims=1
            )
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
            
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
            
        return float(loss)
    
    def _sample_batch(self):
        """Sample a batch of experiences from memory."""
        indices = np.random.randint(len(self.memory), size=self.batch_size)
        batch = [self.memory[idx] for idx in indices]
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones
    
    def update_target_network(self):
        """Update target network weights with current Q-network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save(self, filepath):
        """Save Q-network to disk."""
        self.q_network.save(filepath)
    
    def load(self, filepath):
        """Load Q-network from disk."""
        self.q_network = tf.keras.models.load_model(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
