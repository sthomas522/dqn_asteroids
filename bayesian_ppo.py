#!/usr/bin/env python3
"""
Bayesian PPO vs Fixed PPO Comparison - Enhanced Version
Now with working video recording and HuggingFace upload functionality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from pathlib import Path
import cv2
from collections import deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
from datetime import datetime

# HuggingFace imports
try:
    from huggingface_hub import HfApi, login, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ HuggingFace Hub not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

class Config:
    def __init__(self):
        self.env_id = "LunarLander-v3"
        self.total_episodes = 3000  # Reduced for faster testing
        self.episodes_per_update = 10
        self.updates_per_epoch = 5
        self.hidden_dim = 256
        
        # Learning parameters
        self.learning_rate = 3e-4
        self.value_learning_rate = 1e-3
        self.ppo_clip_epsilon = 0.2
        self.ppo_entropy_coef = 0.01
        self.ppo_value_coef = 0.5
        
        # Bayesian parameters
        self.acceptance_threshold = 0.01
        self.fallback_clip = 0.1
        self.param_penalty = 0.0005
        
        # GAE parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Training parameters
        self.max_grad_norm = 0.5
        self.normalize_advantages = True
        self.advantage_eps = 1e-8
        
        # Visualization
        self.plot_frequency = 200
        self.video_frequency = 500
        
        # HuggingFace settings
        self.hf_repo_id = None  # Will be set during upload
        self.hf_username = None  # Will be set during upload

class Network(nn.Module):
    """Shared network architecture for both algorithms"""
    
    def __init__(self, config, n_obs=8, n_actions=4):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(n_obs, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, n_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Policy head small initialization
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_head[-1].bias, 0)
        
        # Value head initialization
        nn.init.xavier_uniform_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, -200.0)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        x = torch.clamp(x, -10, 10)
        
        shared = self.shared_features(x)
        policy_logits = self.policy_head(shared)
        value = self.value_head(shared)
        
        return policy_logits, value.squeeze(-1)
    
    def get_action_and_value(self, state):
        with torch.no_grad():
            logits, value = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action.item(), log_prob.item(), value.item(), entropy.item()
    
    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

class FixedPPO:
    """Standard PPO with clipping"""
    
    def __init__(self, network, config):
        self.network = network
        self.config = config
        
        # Separate optimizers
        policy_params = list(network.shared_features.parameters()) + list(network.policy_head.parameters())
        value_params = list(network.shared_features.parameters()) + list(network.value_head.parameters())
        
        self.policy_optimizer = optim.Adam(policy_params, lr=config.learning_rate)
        self.value_optimizer = optim.Adam(value_params, lr=config.value_learning_rate)
        
        # Statistics tracking
        self.policy_losses = []
        self.value_losses = []
        self.approx_kls = []
        self.clip_fractions = []
        self.explained_variances = []
    
    def compute_advantages_and_returns(self, rewards, values, dones):
        """Compute GAE advantages and returns"""
        advantages = []
        returns = []
        gae = 0
        
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        
        values = list(values) + [0.0]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t] 
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        if self.config.normalize_advantages and len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            if adv_std > self.config.advantage_eps:
                advantages = (advantages - adv_mean) / (adv_std + self.config.advantage_eps)
                advantages = np.clip(advantages, -5, 5)
        
        return advantages, returns
    
    def update(self, states, actions, advantages, old_log_probs, returns, old_values):
        """Standard PPO update with clipping"""
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        old_values = torch.FloatTensor(old_values)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        
        for epoch in range(self.config.updates_per_epoch):
            # Forward pass
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
            
            # === VALUE FUNCTION UPDATE ===
            value_loss = 0.5 * torch.mean((new_values - returns) ** 2)
            
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.value_head.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
            
            # === POLICY UPDATE ===
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
            
            # Standard PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 1 + self.config.ppo_clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Entropy bonus
            entropy_loss = -torch.mean(entropy)
            
            # Combined policy loss
            total_policy_objective = policy_loss + self.config.ppo_entropy_coef * entropy_loss
            
            self.policy_optimizer.zero_grad()
            total_policy_objective.backward()
            torch.nn.utils.clip_grad_norm_(self.network.policy_head.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            # Statistics
            with torch.no_grad():
                approx_kl = torch.mean((old_log_probs - new_log_probs) ** 2) / 2
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.config.ppo_clip_epsilon).float())
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
            
            # Early stopping on high KL
            if approx_kl > 0.01:
                break
        
        # Average statistics
        num_epochs = epoch + 1
        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        avg_approx_kl = total_approx_kl / num_epochs
        avg_clip_fraction = total_clip_fraction / num_epochs
        
        # Explained variance
        with torch.no_grad():
            explained_var = 1 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)
        
        # Store statistics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.approx_kls.append(avg_approx_kl)
        self.clip_fractions.append(avg_clip_fraction)
        self.explained_variances.append(explained_var.item())
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'approx_kl': avg_approx_kl,
            'clip_fraction': avg_clip_fraction,
            'explained_variance': explained_var.item(),
            'epochs_completed': num_epochs
        }

class BayesianPPO:
    """Improved Bayesian PPO with better acceptance logic"""
    
    def __init__(self, network, config):
        self.network = network
        self.config = config
        
        # Separate optimizers like Fixed PPO for better stability
        policy_params = list(network.shared_features.parameters()) + list(network.policy_head.parameters())
        value_params = list(network.shared_features.parameters()) + list(network.value_head.parameters())
        
        self.policy_optimizer = optim.Adam(policy_params, lr=config.learning_rate)
        self.value_optimizer = optim.Adam(value_params, lr=config.value_learning_rate)
        
        # Statistics tracking
        self.policy_losses = []
        self.value_losses = []
        self.log_posteriors = []
        self.acceptance_rates = []
        self.explained_variances = []
        
        # Efficient tracking with longer history for stability
        self.recent_losses = deque(maxlen=20)  # Longer history
        self.update_count = 0
        
        print(f"🧠 Improved Bayesian PPO initialized successfully!")
    
    def compute_advantages_and_returns(self, rewards, values, dones):
        """Compute GAE advantages and returns"""
        advantages = []
        returns = []
        gae = 0
        
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        
        values = list(values) + [0.0]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t] 
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        if self.config.normalize_advantages and len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            if adv_std > self.config.advantage_eps:
                advantages = (advantages - adv_mean) / (adv_std + self.config.advantage_eps)
                advantages = np.clip(advantages, -5, 5)
        
        return advantages, returns
    
    def compute_acceptance_probability(self, current_loss, baseline_loss):
        """Improved acceptance decision - more permissive for learning"""
        if baseline_loss == float('inf'):
            return 0.9  # Accept first update with high probability
            
        improvement = baseline_loss - current_loss  # Positive is good
        
        # More permissive acceptance logic for better learning
        if improvement > 0.005:  # Lower threshold for "good" improvement
            return min(0.95, 0.8 + improvement * 20)  # High acceptance
        elif improvement > -0.01:  # Accept small degradation
            return max(0.6, 0.7 + improvement * 50)  # Still reasonable acceptance
        else:  # Larger degradation
            return max(0.3, 0.5 * np.exp(improvement * 2))  # More permissive than before
    
    def update(self, states, actions, advantages, old_log_probs, returns, old_values):
        """Efficient Bayesian PPO update"""
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        old_values = torch.FloatTensor(old_values)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_acceptance = 0
        
        # Baseline loss for comparison
        baseline_loss = np.mean(self.recent_losses) if self.recent_losses else float('inf')
        
        for epoch in range(self.config.updates_per_epoch):
            # === VALUE FUNCTION UPDATE (SEPARATE LIKE FIXED PPO) ===
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
            value_loss = 0.5 * torch.mean((new_values - returns) ** 2)
            
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.value_head.parameters(), self.config.max_grad_norm)
            self.value_optimizer.step()
            
            # === POLICY UPDATE WITH BAYESIAN ACCEPTANCE ===
            new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
            
            # Standard PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 1 + self.config.ppo_clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Entropy loss
            entropy_loss = -torch.mean(entropy)
            
            # Combined policy objective
            policy_objective = policy_loss + self.config.ppo_entropy_coef * entropy_loss
            
            # Bayesian acceptance decision based on policy loss only
            current_loss_val = policy_loss.item()
            acceptance_prob = self.compute_acceptance_probability(current_loss_val, baseline_loss)
            accept = np.random.random() < acceptance_prob
            
            if accept:
                # Apply the policy update
                self.policy_optimizer.zero_grad()
                policy_objective.backward()
                torch.nn.utils.clip_grad_norm_(self.network.policy_head.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update baseline
                self.recent_losses.append(current_loss_val)
                total_acceptance += 1
                
                if epoch == 0:
                    print(f"  ✅ Accepted (α={acceptance_prob:.3f}) policy_loss: {current_loss_val:.3f}")
            else:
                # Skip the policy update (but value update still happened)
                if epoch == 0:
                    print(f"  ❌ Rejected (α={acceptance_prob:.3f}) policy_loss: {current_loss_val:.3f}")
            
            # Statistics (always track both)
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Average statistics
        num_epochs = self.config.updates_per_epoch
        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        acceptance_rate = total_acceptance / num_epochs
        avg_log_posterior = -np.mean(self.recent_losses) if self.recent_losses else 0
        
        # Explained variance
        with torch.no_grad():
            explained_var = 1 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)
        
        # Store statistics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.log_posteriors.append(avg_log_posterior)
        self.acceptance_rates.append(acceptance_rate)
        self.explained_variances.append(explained_var.item())
        
        self.update_count += 1
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'log_posterior': avg_log_posterior,
            'acceptance_rate': acceptance_rate,
            'explained_variance': explained_var.item()
        }

def collect_trajectory(env, network, max_steps=1000):
    """Collect single trajectory"""
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    
    state, _ = env.reset()
    total_reward = 0
    step = 0
    
    while step < max_steps:
        action, log_prob, value, entropy = network.get_action_and_value(state)
        
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        dones.append(done)
        total_reward += reward
        
        state = next_state
        step += 1
        
        if done:
            break
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'log_probs': np.array(log_probs),
        'values': np.array(values),
        'dones': np.array(dones),
        'total_reward': total_reward,
        'length': len(rewards)
    }

def record_video(env, network, video_path, max_steps=1000):
    """Record a video of the agent playing - FIXED VERSION"""
    print(f"🎬 Recording video to {video_path}")
    
    # Ensure the environment supports rgb_array rendering
    try:
        # Create a new environment specifically for recording
        if hasattr(env, 'spec') and env.spec is not None:
            env_id = env.spec.id
        else:
            env_id = "LunarLander-v3"  # fallback
        
        # Create fresh environment with render mode
        recording_env = gym.make(env_id, render_mode="rgb_array")
        
        frames = []
        state, _ = recording_env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Render and capture frame
            frame = recording_env.render()
            if frame is not None:
                frames.append(frame)
            
            # Get action from network
            action, _, _, _ = network.get_action_and_value(state)
            state, reward, terminated, truncated, _ = recording_env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        recording_env.close()
        
        # Save video if we have frames
        if frames and len(frames) > 0:
            # Ensure output directory exists
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            
            video_writer = cv2.VideoWriter(
                str(video_path), 
                fourcc, 
                fps, 
                (width, height)
            )
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            video_writer.release()
            
            print(f"✅ Video saved: {video_path} ({len(frames)} frames, reward: {total_reward:.1f})")
            return total_reward
        else:
            print("❌ No frames captured for video")
            return 0
            
    except Exception as e:
        print(f"❌ Video recording failed: {e}")
        return 0

def save_model_for_hf(network, config, results, save_dir):
    """Save model and metadata for HuggingFace upload"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    torch.save({
        'model_state_dict': network.state_dict(),
        'config': config.__dict__,
        'results': results
    }, save_dir / "model.pth")
    
    # Create model card
    model_card = f"""---
tags:
- LunarLander-v3
- ppo
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
library_name: stable-baselines3
---

# PPO Agent playing LunarLander-v3

This is a **PPO** agent trained on the **LunarLander-v3** environment.

## Usage

```python
import torch
import gymnasium as gym
from pathlib import Path

# Load the model
checkpoint = torch.load("model.pth")
network = Network(config)  # You need to define the Network class
network.load_state_dict(checkpoint['model_state_dict'])

# Test the agent
env = gym.make("LunarLander-v3")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _, _, _ = network.get_action_and_value(state)
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {{total_reward}}")
```

## Training Results

- **Environment**: LunarLander-v3
- **Training Episodes**: {len(results.get('rewards', []))}
- **Final Performance**: {np.mean(results['rewards'][-100:]) if len(results.get('rewards', [])) >= 100 else 'N/A':.1f} ± {np.std(results['rewards'][-100:]) if len(results.get('rewards', [])) >= 100 else 'N/A':.1f}
- **Best Episode**: {max(results['rewards']) if results.get('rewards') else 'N/A'}

## Algorithm Details

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Network Architecture**: Actor-Critic with shared features
- **Learning Rate**: {config.learning_rate}
- **Clip Epsilon**: {config.ppo_clip_epsilon}
- **Training Episodes**: {config.total_episodes}

"""
    
    with open(save_dir / "README.md", 'w') as f:
        f.write(model_card)
    
    # Save hyperparameters
    hyperparams = {
        "env_id": config.env_id,
        "learning_rate": config.learning_rate,
        "ppo_clip_epsilon": config.ppo_clip_epsilon,
        "ppo_entropy_coef": config.ppo_entropy_coef,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "hidden_dim": config.hidden_dim,
        "total_episodes": config.total_episodes,
        "episodes_per_update": config.episodes_per_update,
        "updates_per_epoch": config.updates_per_epoch
    }
    
    with open(save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Save training metrics
    metrics = {
        "episode_rewards": results.get('rewards', []),
        "episode_lengths": results.get('lengths', []),
        "policy_losses": results.get('policy_losses', []),
        "value_losses": results.get('value_losses', []),
        "explained_variances": results.get('explained_variances', [])
    }
    
    with open(save_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Model saved to {save_dir}")
    return save_dir

def upload_to_huggingface(model_dir, repo_name, private=False):
    """Upload trained model to HuggingFace Hub"""
    if not HF_AVAILABLE:
        print("❌ HuggingFace Hub not available. Please install: pip install huggingface_hub")
        return None
    
    try:
        # Login to HuggingFace
        print("🔑 Please log in to HuggingFace...")
        login()
        
        # Get username
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        
        repo_id = f"{username}/{repo_name}"
        print(f"📤 Uploading to {repo_id}")
        
        # Create repository
        try:
            create_repo(repo_id, private=private, exist_ok=True)
            print(f"✅ Repository {repo_id} created/verified")
        except Exception as e:
            print(f"⚠️ Repository creation issue (may already exist): {e}")
        
        # Upload files
        model_dir = Path(model_dir)
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                print(f"📁 Uploading {file_path.name}...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="model"
                )
        
        print(f"🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")
        return repo_id
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def run_single_algorithm(algorithm_name, algorithm_class, network_class, config, record_videos=True):
    """Run a single algorithm and return results"""
    print(f"\n🚀 Starting {algorithm_name}...")
    
    try:
        network = network_class(config)
        algorithm = algorithm_class(network, config)
        
        # Create output directory
        results_dir = Path(f"{algorithm_name.lower().replace(' ', '_')}_results")
        results_dir.mkdir(exist_ok=True)
        
        # Create videos subdirectory
        videos_dir = results_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # Results tracking
        results = {
            'rewards': [], 'lengths': [],
            'policy_losses': [], 'value_losses': [], 'explained_variances': []
        }
        
        # Add algorithm-specific metrics
        if hasattr(algorithm, 'log_posteriors'):
            results['log_posteriors'] = []
            results['acceptance_rates'] = []
        if hasattr(algorithm, 'approx_kls'):
            results['approx_kls'] = []
            results['clip_fractions'] = []
        
        episode_count = 0
        start_time = time.time()
        
        while episode_count < config.total_episodes:
            # Collect trajectories
            env = gym.make(config.env_id)
            trajectories = []
            batch_rewards = []
            
            for ep in range(config.episodes_per_update):
                if episode_count >= config.total_episodes:
                    break
                
                traj = collect_trajectory(env, network)
                trajectories.append(traj)
                results['rewards'].append(traj['total_reward'])
                results['lengths'].append(traj['length'])
                batch_rewards.append(traj['total_reward'])
                episode_count += 1
            
            env.close()
            
            # Record video periodically
            if record_videos and episode_count % config.video_frequency == 0:
                video_path = videos_dir / f"episode_{episode_count}.mp4"
                env_for_video = gym.make(config.env_id)
                video_reward = record_video(env_for_video, network, video_path)
                env_for_video.close()
                print(f"📹 Video recorded at episode {episode_count}, reward: {video_reward:.1f}")
            
            if trajectories:
                # Process data
                all_states = np.concatenate([t['states'] for t in trajectories])
                all_actions = np.concatenate([t['actions'] for t in trajectories])
                all_rewards = np.concatenate([t['rewards'] for t in trajectories])
                all_log_probs = np.concatenate([t['log_probs'] for t in trajectories])
                all_values = np.concatenate([t['values'] for t in trajectories])
                all_dones = np.concatenate([t['dones'] for t in trajectories])
                
                # Compute advantages
                advantages, returns = algorithm.compute_advantages_and_returns(
                    all_rewards, all_values, all_dones
                )
                
                # Algorithm update
                stats = algorithm.update(
                    all_states, all_actions, advantages, 
                    all_log_probs, returns, all_values
                )
                
                # Store stats
                results['policy_losses'].append(stats['policy_loss'])
                results['value_losses'].append(stats['value_loss'])
                results['explained_variances'].append(stats['explained_variance'])
                
                if 'log_posterior' in stats:
                    results['log_posteriors'].append(stats['log_posterior'])
                    results['acceptance_rates'].append(stats['acceptance_rate'])
                if 'approx_kl' in stats:
                    results['approx_kls'].append(stats['approx_kl'])
                    results['clip_fractions'].append(stats['clip_fraction'])
            
            # Progress update
            if episode_count % 100 == 0:
                recent = results['rewards'][-50:] if len(results['rewards']) >= 50 else results['rewards']
                print(f"{algorithm_name} - Episode {episode_count}: {np.mean(recent):.1f} ± {np.std(recent):.1f}")
        
        # Record final video
        if record_videos:
            final_video_path = videos_dir / "final_performance.mp4"
            env_for_video = gym.make(config.env_id)
            final_reward = record_video(env_for_video, network, final_video_path)
            env_for_video.close()
            print(f"🎬 Final video recorded, reward: {final_reward:.1f}")
        
        # Save results
        with open(results_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save model for potential HuggingFace upload
        if algorithm_name.lower() == 'fixed ppo':  # Only save Fixed PPO for HF
            model_save_dir = save_model_for_hf(network, config, results, results_dir / "hf_model")
            
            # Ask user if they want to upload to HuggingFace
            if HF_AVAILABLE:
                print(f"\n🤗 Would you like to upload your {algorithm_name} model to HuggingFace?")
                upload_choice = input("Enter 'y' to upload, or any other key to skip: ").strip().lower()
                
                if upload_choice == 'y':
                    repo_name = input("Enter repository name (e.g., 'ppo-lunarlander'): ").strip()
                    if repo_name:
                        private_choice = input("Make repository private? (y/n): ").strip().lower()
                        is_private = private_choice == 'y'
                        
                        repo_id = upload_to_huggingface(model_save_dir, repo_name, private=is_private)
                        if repo_id:
                            print(f"🎉 Model uploaded! View at: https://huggingface.co/{repo_id}")
                        else:
                            print("❌ Upload failed")
                    else:
                        print("❌ No repository name provided, skipping upload")
                else:
                    print("⏭️ Skipping HuggingFace upload")
        
        total_time = time.time() - start_time
        print(f"✅ {algorithm_name} completed in {total_time/60:.1f} minutes")
        
        return results
        
    except Exception as e:
        print(f"❌ {algorithm_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'rewards': [], 'lengths': [], 'policy_losses': [], 'value_losses': [], 'explained_variances': []}

def create_comparison_plots(fixed_results, bayesian_results):
    """Create side-by-side comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards comparison
    ax = axes[0, 0]
    if fixed_results['rewards']:
        episodes = range(len(fixed_results['rewards']))
        ax.plot(episodes, fixed_results['rewards'], 'b-', alpha=0.3, linewidth=0.5, label='Fixed PPO')
        
        if len(fixed_results['rewards']) > 20:
            window = 50
            smooth = np.convolve(fixed_results['rewards'], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(fixed_results['rewards'])), smooth, 'b-', linewidth=3)
    
    if bayesian_results['rewards']:
        episodes = range(len(bayesian_results['rewards']))
        ax.plot(episodes, bayesian_results['rewards'], 'r-', alpha=0.3, linewidth=0.5, label='Bayesian PPO')
        
        if len(bayesian_results['rewards']) > 20:
            window = 50
            smooth = np.convolve(bayesian_results['rewards'], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(bayesian_results['rewards'])), smooth, 'r-', linewidth=3)
    
    ax.axhline(y=200, color='g', linestyle='--', alpha=0.7, label='Success (200)')
    ax.set_title('Learning Curves Comparison')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Acceptance rate
    ax = axes[0, 1]
    if bayesian_results.get('acceptance_rates'):
        updates = range(len(bayesian_results['acceptance_rates']))
        ax.plot(updates, bayesian_results['acceptance_rates'], 'purple', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Target (0.5)')
        ax.set_title('Bayesian PPO: Acceptance Rate')
        ax.set_xlabel('Update')
        ax.set_ylabel('Acceptance Rate')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value losses
    ax = axes[0, 2]
    if fixed_results['value_losses']:
        updates = range(len(fixed_results['value_losses']))
        ax.plot(updates, fixed_results['value_losses'], 'b-', linewidth=2, label='Fixed PPO')
    if bayesian_results['value_losses']:
        updates = range(len(bayesian_results['value_losses']))
        ax.plot(updates, bayesian_results['value_losses'], 'r-', linewidth=2, label='Bayesian PPO')
    ax.set_title('Value Loss Comparison')
    ax.set_xlabel('Update')
    ax.set_ylabel('Value Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance distributions
    ax = axes[1, 0]
    if fixed_results['rewards']:
        recent_fixed = fixed_results['rewards'][-200:] if len(fixed_results['rewards']) >= 200 else fixed_results['rewards']
        ax.hist(recent_fixed, bins=30, alpha=0.6, label='Fixed PPO', color='blue')
    if bayesian_results['rewards']:
        recent_bayesian = bayesian_results['rewards'][-200:] if len(bayesian_results['rewards']) >= 200 else bayesian_results['rewards']
        ax.hist(recent_bayesian, bins=30, alpha=0.6, label='Bayesian PPO', color='red')
    ax.axvline(x=200, color='g', linestyle='--', alpha=0.7, label='Success threshold')
    ax.set_title('Recent Performance Distribution')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Explained variance
    ax = axes[1, 1]
    if fixed_results['explained_variances']:
        updates = range(len(fixed_results['explained_variances']))
        ax.plot(updates, fixed_results['explained_variances'], 'b-', linewidth=2, label='Fixed PPO')
    if bayesian_results['explained_variances']:
        updates = range(len(bayesian_results['explained_variances']))
        ax.plot(updates, bayesian_results['explained_variances'], 'r-', linewidth=2, label='Bayesian PPO')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title('Explained Variance Comparison')
    ax.set_xlabel('Update')
    ax.set_ylabel('Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "ALGORITHM COMPARISON\n\n"
    
    if fixed_results['rewards']:
        recent_fixed = fixed_results['rewards'][-100:] if len(fixed_results['rewards']) >= 100 else fixed_results['rewards']
        success_rate_fixed = np.mean(np.array(recent_fixed) > 200) * 100
        summary_text += f"Fixed PPO:\n"
        summary_text += f"  Final: {np.mean(recent_fixed):.1f} ± {np.std(recent_fixed):.1f}\n"
        summary_text += f"  Best: {np.max(fixed_results['rewards']):.1f}\n"
        summary_text += f"  Success Rate: {success_rate_fixed:.1f}%\n\n"
    
    if bayesian_results['rewards']:
        recent_bayesian = bayesian_results['rewards'][-100:] if len(bayesian_results['rewards']) >= 100 else bayesian_results['rewards']
        success_rate_bayesian = np.mean(np.array(recent_bayesian) > 200) * 100
        summary_text += f"Bayesian PPO:\n"
        summary_text += f"  Final: {np.mean(recent_bayesian):.1f} ± {np.std(recent_bayesian):.1f}\n"
        summary_text += f"  Best: {np.max(bayesian_results['rewards']):.1f}\n"
        summary_text += f"  Success Rate: {success_rate_bayesian:.1f}%\n"
        
        if bayesian_results.get('acceptance_rates'):
            avg_acceptance = np.mean(bayesian_results['acceptance_rates'])
            summary_text += f"  Avg Acceptance: {avg_acceptance:.3f}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_comparison_experiment():
    """Run both algorithms sequentially and compare results"""
    print("🔬 ENHANCED BAYESIAN PPO vs FIXED PPO COMPARISON")
    print("="*60)
    
    config = Config()
    
    # Run Fixed PPO first
    print("\n1️⃣ Running Fixed PPO...")
    fixed_results = run_single_algorithm('Fixed PPO', FixedPPO, Network, config, record_videos=True)
    
    # Run Bayesian PPO second
    print("\n2️⃣ Running Bayesian PPO...")
    bayesian_results = run_single_algorithm('Bayesian PPO', BayesianPPO, Network, config, record_videos=True)
    
    print(f"\n🏁 BOTH ALGORITHMS COMPLETED!")
    
    # Create comparison plots
    if fixed_results['rewards'] or bayesian_results['rewards']:
        create_comparison_plots(fixed_results, bayesian_results)
        
        # Final summary
        print(f"\n🎯 FINAL COMPARISON SUMMARY:")
        if fixed_results['rewards']:
            recent_fixed = fixed_results['rewards'][-100:] if len(fixed_results['rewards']) >= 100 else fixed_results['rewards']
            success_rate_fixed = np.mean(np.array(recent_fixed) > 200) * 100
            print(f"  Fixed PPO: {np.mean(recent_fixed):.1f}±{np.std(recent_fixed):.1f} (success: {success_rate_fixed:.1f}%)")
        
        if bayesian_results['rewards']:
            recent_bayesian = bayesian_results['rewards'][-100:] if len(bayesian_results['rewards']) >= 100 else bayesian_results['rewards']
            success_rate_bayesian = np.mean(np.array(recent_bayesian) > 200) * 100
            print(f"  Bayesian PPO: {np.mean(recent_bayesian):.1f}±{np.std(recent_bayesian):.1f} (success: {success_rate_bayesian:.1f}%)")
            
            if bayesian_results.get('acceptance_rates'):
                avg_acceptance = np.mean(bayesian_results['acceptance_rates'])
                print(f"  Average acceptance rate: {avg_acceptance:.3f}")
    
    return {'Fixed PPO': fixed_results, 'Bayesian PPO': bayesian_results}

def run_single_experiment(algorithm_type='bayesian'):
    """Run a single algorithm experiment"""
    config = Config()
    
    if algorithm_type.lower() == 'bayesian':
        print("🧠 Running Bayesian PPO experiment...")
        return run_single_algorithm('Bayesian PPO', BayesianPPO, Network, config, record_videos=True)
    else:
        print("🔧 Running Fixed PPO experiment...")
        return run_single_algorithm('Fixed PPO', FixedPPO, Network, config, record_videos=True)

def test_video_recording():
    """Test video recording functionality"""
    print("🎬 Testing video recording...")
    
    config = Config()
    network = Network(config)
    
    # Create test environment
    env = gym.make(config.env_id)
    
    # Record a short test video
    test_video_path = Path("test_video.mp4")
    reward = record_video(env, network, test_video_path, max_steps=100)
    
    env.close()
    
    if test_video_path.exists():
        print(f"✅ Video recording works! Test video saved with reward: {reward:.1f}")
        # Clean up test file
        test_video_path.unlink()
    else:
        print("❌ Video recording failed")

def demo_hf_upload():
    """Demo HuggingFace upload process with a dummy model"""
    if not HF_AVAILABLE:
        print("❌ HuggingFace Hub not available. Install with: pip install huggingface_hub")
        return
    
    print("🤗 Demo HuggingFace upload process...")
    
    # Create a dummy model and results
    config = Config()
    network = Network(config)
    
    # Dummy results
    dummy_results = {
        'rewards': [100, 150, 200, 250] * 25,  # 100 episodes
        'lengths': [200] * 100,
        'policy_losses': [0.1] * 25,
        'value_losses': [0.05] * 25,
        'explained_variances': [0.8] * 25
    }
    
    # Save model
    model_dir = save_model_for_hf(network, config, dummy_results, "demo_hf_model")
    
    print("🎯 Files prepared for HuggingFace upload:")
    for file_path in model_dir.iterdir():
        print(f"  📁 {file_path.name}")
    
    print("\n🚀 To upload, use: upload_to_huggingface('demo_hf_model', 'your-repo-name')")

def upload_existing_model(model_path, repo_name="ppo2-LunarLander-v2", private=False):
    """Upload an existing trained model to HuggingFace"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model path {model_path} does not exist")
        return None
    
    print(f"🚀 Uploading existing model from {model_path} to {repo_name}")
    
    if model_path.is_file() and model_path.suffix == '.pkl':
        # Load results from pickle file
        with open(model_path, 'rb') as f:
            results = pickle.load(f)
        
        # Look for model.pth in same directory or parent directory
        model_dir = model_path.parent
        model_file = model_dir / "hf_model" / "model.pth"
        
        if not model_file.exists():
            print(f"❌ Could not find model.pth at {model_file}")
            print("Available files:")
            for f in model_dir.rglob("*"):
                if f.is_file():
                    print(f"  {f}")
            return None
            
        # Load the model
        checkpoint = torch.load(model_file)
        config_dict = checkpoint['config']
        
        # Recreate config object
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # Recreate network
        network = Network(config)
        network.load_state_dict(checkpoint['model_state_dict'])
        
        # Save for HF upload
        hf_model_dir = save_model_for_hf(network, config, results, "upload_temp")
        
    elif model_path.is_dir():
        # Assume it's already a prepared HF model directory
        hf_model_dir = model_path
    else:
        print(f"❌ Invalid model path: {model_path}")
        return None
    
    # Upload to HuggingFace
    repo_id = upload_to_huggingface(hf_model_dir, repo_name, private=private)
    
    # Clean up temp directory if created
    if model_path.is_file() and Path("upload_temp").exists():
        import shutil
        shutil.rmtree("upload_temp")
    
    return repo_id

def quick_upload_to_specific_repo():
    """Quick function to upload to ppo2-LunarLander-v2 specifically"""
    print("🎯 Quick upload to ppo2-LunarLander-v2")
    
    # Look for existing Fixed PPO results
    fixed_ppo_dir = Path("fixed_ppo_results")
    
    if not fixed_ppo_dir.exists():
        print("❌ No Fixed PPO results found. Run training first:")
        print("   python script.py fixed")
        return None
    
    # Find the model files
    hf_model_dir = fixed_ppo_dir / "hf_model"
    results_file = fixed_ppo_dir / "results.pkl"
    
    if hf_model_dir.exists() and results_file.exists():
        print("✅ Found existing Fixed PPO model!")
        
        # Confirm upload
        confirm = input(f"Upload to ppo2-LunarLander-v2? (y/n): ").strip().lower()
        if confirm == 'y':
            private = input("Make repository private? (y/n): ").strip().lower() == 'y'
            repo_id = upload_to_huggingface(hf_model_dir, "ppo2-LunarLander-v2", private=private)
            
            if repo_id:
                print(f"🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")
                return repo_id
            else:
                print("❌ Upload failed")
                return None
        else:
            print("⏭️ Upload cancelled")
            return None
    else:
        print("❌ Model files not found. Available files:")
        for f in fixed_ppo_dir.rglob("*"):
            if f.is_file():
                print(f"  {f}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            run_comparison_experiment()
        elif sys.argv[1] == "bayesian":
            results = run_single_experiment('bayesian')
        elif sys.argv[1] == "fixed":
            results = run_single_experiment('fixed')
        elif sys.argv[1] == "test_video":
            test_video_recording()
        elif sys.argv[1] == "demo_hf":
            demo_hf_upload()
        elif sys.argv[1] == "upload":
            # Quick upload to your specific repo
            quick_upload_to_specific_repo()
        elif sys.argv[1] == "upload_existing" and len(sys.argv) > 2:
            # Upload existing model: python script.py upload_existing path/to/model
            model_path = sys.argv[2]
            repo_name = sys.argv[3] if len(sys.argv) > 3 else "ppo2-LunarLander-v2"
            upload_existing_model(model_path, repo_name)
        else:
            print("Usage: python script.py [compare|bayesian|fixed|test_video|demo_hf|upload|upload_existing]")
            print("  upload                    - Quick upload existing Fixed PPO to ppo2-LunarLander-v2")
            print("  upload_existing <path>    - Upload existing model from path")
    else:
        # Default: run comparison
        run_comparison_experiment()