import tensorflow as tf
import numpy as np
from .base import DiscreteActionAgent

class PolicyGradientAgent(DiscreteActionAgent):
    """Policy Gradient (REINFORCE) agent implementation."""
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=1e-3,
                 gamma=0.99):
        """Initialize Policy Gradient agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            gamma: Discount factor
        """
        super().__init__(state_dim, action_dim)
        
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def build_networks(self):
        """Build policy network."""
        self.policy_network = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=self.state_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
    
    def get_action(self, state, training=True):
        """Get action by sampling from policy network."""
        probs = self.policy_network(state[np.newaxis])[0]
        if training:
            return np.random.choice(self.action_dim, p=probs.numpy())
        else:
            return tf.argmax(probs).numpy()
    
    def store_episode_experience(self, state, action, reward):
        """Store experience from one step of an episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_step(self):
        """Train on the collected experience after episode ends."""
        # Convert episode data to tensors
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        
        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(self.rewards)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - tf.reduce_mean(discounted_rewards)) / \
                           (tf.math.reduce_std(discounted_rewards) + 1e-8)
        
        with tf.GradientTape() as tape:
            # Get probabilities of actions taken
            probs = self.policy_network(states)
            indices = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + actions
            action_probs = tf.gather(tf.reshape(probs, [-1]), indices)
            
            # Calculate loss
            log_probs = tf.math.log(action_probs)
            loss = -tf.reduce_mean(log_probs * discounted_rewards)
        
        # Apply gradients
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return float(loss)
    
    def _discount_rewards(self, rewards):
        """Calculate discounted rewards."""
        discounted = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted[t] = running_add
        return discounted
    
    def save(self, filepath):
        """Save policy network to disk."""
        self.policy_network.save(filepath)
    
    def load(self, filepath):
        """Load policy network from disk."""
        self.policy_network = tf.keras.models.load_model(filepath)
