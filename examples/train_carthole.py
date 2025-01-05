import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.dqn import DQNAgent
from agents.policy_gradient import PolicyGradientAgent

def plot_training_results(rewards, ma_rewards, title):
    """Plot training rewards and moving average."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards', alpha=0.5)
    plt.plot(ma_rewards, label='Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def train_dqn(env, episodes=500, render_freq=100):
    """Train DQN agent on environment."""
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=10000,
        memory_size=10000,
        batch_size=64,
        target_update_freq=100,
        double_dqn=True,
        dueling=True
    )
    
    # Training loop
    rewards = []
    ma_rewards = []  # Moving average rewards
    best_reward = -np.inf
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            
            total_reward += reward
            state = next_state
            
            # Render every render_freq episodes
            if episode % render_freq == 0:
                env.render()
        
        rewards.append(total_reward)
        ma_rewards.append(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('best_dqn_model.h5')
        
        print(f"Episode {episode+1}/{episodes}, " 
              f"Reward: {total_reward:.1f}, "
              f"Moving Avg: {ma_rewards[-1]:.1f}, "
              f"Epsilon: {agent._get_epsilon():.3f}")
        
    return rewards, ma_rewards

def train_policy_gradient(env, episodes=500, render_freq=100):
    """Train Policy Gradient agent on environment."""
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = PolicyGradientAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99
    )
    
    # Training loop
    rewards = []
    ma_rewards = []
    best_reward = -np.inf
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        # Collect experience for one episode
        while not (done or truncated):
            # Get action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.store_episode_experience(state, action, reward)
            
            total_reward += reward
            state = next_state
            
            # Render every render_freq episodes
            if episode % render_freq == 0:
                env.render()
        
        # Train on collected experience
        loss = agent.train_step()
        
        rewards.append(total_reward)
        ma_rewards.append(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('best_pg_model.h5')
        
        print(f"Episode {episode+1}/{episodes}, "
              f"Reward: {total_reward:.1f}, "
              f"Moving Avg: {ma_rewards[-1]:.1f}")
    
    return rewards, ma_rewards

def main():
    # Create environment
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    env.reset(seed=42)
    
    # Training parameters
    EPISODES = 500
    RENDER_FREQ = 100
    
    # Train agents
    print("Training DQN agent...")
    dqn_rewards, dqn_ma = train_dqn(env, EPISODES, RENDER_FREQ)
    plot_training_results(dqn_rewards, dqn_ma, "DQN Training Results")
    
    print("\nTraining Policy Gradient agent...")
    pg_rewards, pg_ma = train_policy_gradient(env, EPISODES, RENDER_FREQ)
    plot_training_results(pg_rewards, pg_ma, "Policy Gradient Training Results")
    
    # Compare results
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_ma, label='DQN', linewidth=2)
    plt.plot(pg_ma, label='Policy Gradient', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('DQN vs Policy Gradient')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    plt.close()
    
    env.close()

if __name__ == "__main__":
    main()
