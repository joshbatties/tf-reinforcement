# TF-Reinforcement

A clean implementation of popular Reinforcement Learning algorithms using TensorFlow 2.x. Features modular design, easy-to-understand implementations, and complete training examples.

## Implemented Algorithms

- **Deep Q-Network (DQN)** with extensions:
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay
- **Policy Gradient** (REINFORCE)

## Features

- Modular and extensible agent architecture
- Full implementations of classic RL environments
- Training visualization and metrics tracking
- Model saving and loading
- Comprehensive training examples

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tf-reinforcement.git
cd tf-reinforcement

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training DQN on CartPole

```bash
python examples/train_cartpole.py
```

This will:
1. Train a DQN agent on the CartPole environment
2. Save training metrics and visualizations
3. Save the best performing model

### Training on LunarLander

```bash
python examples/train_lunarlander.py
```

## Examples

### Using DQN Agent

```python
from agents.dqn import DQNAgent
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Initialize agent
agent = DQNAgent(
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.n,
    learning_rate=0.001,
    gamma=0.99
)

# Training loop
for episode in range(500):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Get action
        action = agent.get_action(state)
        
        # Take step
        next_state, reward, done, _, _ = env.step(action)
        
        # Store experience and train
        agent.store_experience(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        state = next_state
```

### Using Policy Gradient Agent

```python
from agents.policy_gradient import PolicyGradientAgent

agent = PolicyGradientAgent(
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.n
)

# Training loop
for episode in range(500):
    state, _ = env.reset()
    done = False
    
    # Collect episode experience
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        agent.store_episode_experience(state, action, reward)
        state = next_state
    
    # Train on episode
    loss = agent.train_step()
```


## Performance

### CartPole-v1
- DQN achieves solving criteria (score > 195) in ~200 episodes
- Policy Gradient solves environment in ~300 episodes

### LunarLander-v2
- DQN achieves solving criteria (score > 200) in ~500 episodes


## Future Work

- [ ] Add A2C implementation
- [ ] Add PPO implementation
- [ ] Add support for continuous action spaces
- [ ] Add more environment examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gymnasium for environments
- DeepMind for DQN architecture references
- Berkeley CS285 for algorithm implementations
