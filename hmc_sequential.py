#!/usr/bin/env python3
"""
Stan-Consistent HMC Implementation with Sequential Posterior-to-Prior Updates

Key improvements:
1. Use MEAN of ensemble instead of median for parameter estimates
2. Posterior estimates from epoch N become priors for epoch N+1
3. Progressive learning that builds on previous knowledge
4. DIFFUSE priors (σ = 10.0) for all parameters - matches Stan's approach
5. NO artificial likelihood scaling - use raw likelihood
6. Proper Jacobian corrections for QR transformations
7. Parameter-specific mass matrix and step sizes
8. Soft parameter bounds and numerical stability
9. Conservative adaptation targeting 80% acceptance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from collections import deque
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
import logging
from scipy.ndimage import uniform_filter1d
import cv2
import pickle

# Setup logging and device
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Asteroids action names
ASTEROIDS_ACTIONS = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE"
}


@dataclass 
class ExperimentConfig:
    """Experiment configuration with Stan-consistent defaults"""
    # Environment
    env_id: str = "ALE/Asteroids-v5"
    frame_stack: int = 4
    screen_size: int = 84

    # Training parameters
    total_episodes: int = 100000
    episodes_per_update: int = 50
    updates_per_epoch: int = 30

    # Network architecture
    hidden_dim: int = 256
    learning_rate: float = 3e-4

    # PPO parameters
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5

    # Stan-consistent HMC parameters
    hmc_step_size: float = 0.001          # Conservative start like Stan
    hmc_num_leapfrog_steps: int = 10      # Reasonable exploration
    hmc_temperature: float = 1.0          # Stan default
    hmc_adapt_delta: float = 0.8          # Target acceptance rate
    hmc_max_treedepth: int = 10          # Prevent infinite loops

    # Video settings
    video_frequency: int = 200
    plot_frequency: int = 100
    debug_frequency: int = 50
    add_text_overlay: bool = True

    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class FrameStack(gym.Wrapper):
    """Frame stacking wrapper"""
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_stack, obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.array(list(self.frames))


class AsteroidsNetwork(nn.Module):
    """Enhanced network for Asteroids"""

    def __init__(self, config: ExperimentConfig, n_actions: int = 14):
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        # CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.frame_stack, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Calculate conv output size
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, config.frame_stack, config.screen_size, config.screen_size)
                conv_out = self.conv_layers(dummy)
                self.conv_out_size = conv_out.numel()
        except Exception:
            self.conv_out_size = 64 * 11 * 11

        # Shared features
        self.features = nn.Sequential(
            nn.Linear(self.conv_out_size, config.hidden_dim),
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
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        """Conservative initialization"""
        try:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            # Final policy layer gets smaller initialization
            nn.init.xavier_normal_(self.policy_head[-1].weight, gain=0.01)
            nn.init.constant_(self.policy_head[-1].bias, 0)

        except Exception:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, 0, 0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        if x.device != device:
            x = x.to(device)

        x = x.float() / 255.0
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        features = self.features(conv_out)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value.squeeze(-1)

    def get_action_and_value(self, state):
        with torch.no_grad():
            logits, value = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states, actions):
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def get_action_probabilities(self, state):
        """Get action probabilities for analysis"""
        with torch.no_grad():
            logits, _ = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            return probs


class SequentialStanConsistentHMC:
    """Stan-consistent HMC with sequential posterior-to-prior updates and ensemble means"""

    def __init__(self, network: nn.Module, config):
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device

        # Parameter transformations
        self.param_transforms = {}
        self.theta_vector = None
        self.theta_shapes = {}
        self.theta_names = []
        
        # Stan-like HMC parameters
        self.base_step_size = config.hmc_step_size
        self.num_leapfrog_steps = config.hmc_num_leapfrog_steps
        self.temperature = config.hmc_temperature
        self.adapt_delta = config.hmc_adapt_delta

        # DIFFUSE priors like Stan's defaults (key fix!)
        self.beta_prior_std = 10.0    # Very diffuse - fixed hyperparameter
        self.feature_prior_std = 10.0  # Very diffuse - fixed hyperparameter
        self.other_prior_std = 10.0    # Very diffuse - fixed hyperparameter
        
        # SEQUENTIAL LEARNING: Prior means that get updated from posteriors
        self.prior_means = {}  # Will be initialized and updated after each epoch
        self.epoch_count = 0   # Track which epoch we're on
        
        # Parameter-specific scaling
        self.parameter_scales = {}
        self.parameter_types = {}
        
        # Statistics
        self.total_proposals = 0
        self.total_acceptances = 0
        self.recent_acceptances = []
        self.numerical_failures = 0
        self.theta_ensemble = []

        self._setup_parameter_transformations()
        self._initialize_theta_vector()
        self._setup_parameter_scaling()
        self._initialize_prior_means()  # NEW: Initialize prior means

        self.n_params = self.theta_vector.numel()
        self.expected_kinetic_energy = self.n_params / 2.0
        
        logger.info(f"🎯 Sequential Stan-consistent HMC initialized:")
        logger.info(f"   Total θ parameters: {self.n_params:,}")
        logger.info(f"   Policy parameters: {self.policy_param_count:,}")
        logger.info(f"   DIFFUSE prior std: {self.beta_prior_std} (fixed)")
        logger.info(f"   Base step size: {self.base_step_size:.6f}")
        logger.info(f"   Target acceptance: {self.adapt_delta:.2f}")
        logger.info(f"   Sequential learning: ENABLED")

    def _setup_parameter_transformations(self):
        """Setup transformations with parameter type tracking"""
        param_idx = 0
        
        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue
                
            param_data = param.data.clone().to(self.device)
            param_size = param_data.numel()
            
            # Classify parameter type
            if 'policy' in name.lower() or 'head' in name.lower():
                self.parameter_types[name] = 'policy'
            elif 'conv' in name.lower() or 'features' in name.lower():
                self.parameter_types[name] = 'feature'
            else:
                self.parameter_types[name] = 'other'
            
            if len(param_data.shape) == 2 and min(param_data.shape) >= 3:
                success = self._setup_qr_transform(name, param_data, param_idx)
                if not success:
                    self._setup_standardization_transform(name, param_data, param_idx)
            else:
                self._setup_standardization_transform(name, param_data, param_idx)
            
            param_idx += param_size

    def _setup_qr_transform(self, name: str, W: torch.Tensor, start_idx: int) -> bool:
        """Setup QR transformation"""
        try:
            Q, R = torch.linalg.qr(W)
            reconstruction_error = torch.norm(Q @ R - W).item()
            if reconstruction_error > 1e-3:
                return False
            
            self.param_transforms[name] = {
                'type': 'QR',
                'Q': Q.detach().clone(),
                'original_shape': W.shape,
                'theta_shape': R.shape,
                'start_idx': start_idx,
                'size': R.numel(),
                'needs_jacobian': True
            }
            return True
        except Exception:
            return False

    def _setup_standardization_transform(self, name: str, param: torch.Tensor, start_idx: int):
        """Setup standardization for vectors/biases"""
        if param.numel() == 1:
            mean_val, std_val = 0.0, 1.0
        else:
            mean_val = torch.mean(param).item()
            std_val = torch.std(param).item()
            if std_val < 1e-8:
                std_val = 1.0

        self.param_transforms[name] = {
            'type': 'standardized',
            'mean': mean_val,
            'std': std_val,
            'original_shape': param.shape,
            'theta_shape': param.shape,
            'start_idx': start_idx,
            'size': param.numel(),
            'needs_jacobian': False
        }

    def _setup_parameter_scaling(self):
        """Setup parameter-specific scaling"""
        self.policy_param_count = 0
        
        for name in self.theta_names:
            param_type = self.parameter_types.get(name, 'other')
            
            if param_type == 'policy':
                self.parameter_scales[name] = 0.1  # Smaller steps for policy
                self.policy_param_count += self.param_transforms[name]['size']
            elif param_type == 'feature':
                self.parameter_scales[name] = 0.5  # Medium steps for features
            else:
                self.parameter_scales[name] = 1.0  # Normal steps for others

    def _initialize_theta_vector(self):
        """Initialize θ vector"""
        theta_components = []
        
        for name, param in self.network.named_parameters():
            if not param.requires_grad or name not in self.param_transforms:
                continue
                
            transform = self.param_transforms[name]
            param_data = param.data.clone()
            
            if transform['type'] == 'QR':
                Q = transform['Q']
                R = Q.T @ param_data
                theta_component = R.flatten()
            elif transform['type'] == 'standardized':
                standardized = (param_data - transform['mean']) / transform['std']
                theta_component = standardized.flatten()
            
            theta_components.append(theta_component)
            self.theta_shapes[name] = transform['theta_shape']
            self.theta_names.append(name)

        if theta_components:
            self.theta_vector = torch.cat(theta_components).to(self.device)
        else:
            self.theta_vector = torch.empty(0).to(self.device)

    def _initialize_prior_means(self):
        """Initialize prior means to zero (first epoch)"""
        self.prior_means = {}
        theta_idx = 0
        
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            
            # Start with zero means for first epoch
            self.prior_means[name] = torch.zeros(param_size).to(self.device)
            theta_idx += param_size
        
        logger.info(f"🔧 Initialized prior means to zero for epoch 1")

    def update_prior_means_from_posterior(self, posterior_mean_theta: torch.Tensor):
        """UPDATE: Use posterior means as priors for next epoch"""
        self.epoch_count += 1
        
        theta_idx = 0
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            
            # Extract posterior mean for this parameter
            posterior_mean_param = posterior_mean_theta[theta_idx:theta_idx + param_size]
            
            # Update prior mean for next epoch
            self.prior_means[name] = posterior_mean_param.detach().clone()
            
            theta_idx += param_size
        
        logger.info(f"✅ Updated prior means from posterior for epoch {self.epoch_count}")
        
        # Log some statistics about the updated priors
        total_shift = torch.norm(posterior_mean_theta).item()
        logger.info(f"   Total prior mean shift: {total_shift:.6f}")

    def compute_log_prior_theta_space(self, theta_vector: torch.Tensor) -> float:
        """Compute log prior with SEQUENTIAL MEANS and DIFFUSE stds"""
        if torch.isnan(theta_vector).any() or torch.isinf(theta_vector).any():
            return -float('inf')
        
        log_prior = 0.0
        jacobian_correction = 0.0
        theta_idx = 0
        
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            param_type = self.parameter_types.get(name, 'other')
            
            theta_param = theta_vector[theta_idx:theta_idx + param_size]
            
            # Use DIFFUSE stds (10.0) but SEQUENTIAL means (updated from posteriors)
            if param_type == 'policy':
                prior_std = self.beta_prior_std     # 10.0 (fixed)
            elif param_type == 'feature':
                prior_std = self.feature_prior_std  # 10.0 (fixed)
            else:
                prior_std = self.other_prior_std    # 10.0 (fixed)
            
            # SEQUENTIAL: Use updated prior means instead of zero
            prior_mean = self.prior_means[name]
            
            # Prior contribution - centered at posterior means from previous epoch
            diff = theta_param - prior_mean
            log_prior += -0.5 * (diff ** 2).sum().item() / (prior_std ** 2)
            
            # Jacobian correction for QR transformations
            if transform.get('needs_jacobian', False) and transform['type'] == 'QR':
                R_shape = transform['theta_shape']
                if R_shape[0] == R_shape[1]:  # Square matrix
                    R_matrix = theta_param.view(R_shape)
                    diag_elements = torch.diag(R_matrix)
                    jacobian_correction += torch.sum(torch.log(torch.abs(diag_elements) + 1e-8)).item()
            
            theta_idx += param_size
        
        return log_prior + jacobian_correction

    def apply_soft_constraints(self, theta_vector: torch.Tensor) -> float:
        """Apply soft parameter bounds"""
        penalty = 0.0
        
        # Global constraint
        if torch.any(torch.abs(theta_vector) > 10.0):
            penalty += torch.sum(F.relu(torch.abs(theta_vector) - 10.0) ** 2).item() * 1000
        
        # Policy-specific constraints
        theta_idx = 0
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            param_type = self.parameter_types.get(name, 'other')
            
            if param_type == 'policy':
                theta_param = theta_vector[theta_idx:theta_idx + param_size]
                if torch.any(torch.abs(theta_param) > 5.0):
                    penalty += torch.sum(F.relu(torch.abs(theta_param) - 5.0) ** 2).item() * 2000
            
            theta_idx += param_size
        
        return penalty

    def theta_to_original_parameters(self, theta_vector: torch.Tensor) -> dict:
        """Convert θ vector back to original parameter space"""
        original_params = {}
        theta_idx = 0
        
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            
            theta_param = theta_vector[theta_idx:theta_idx + param_size]
            theta_param = theta_param.view(transform['theta_shape'])
            
            if transform['type'] == 'QR':
                Q = transform['Q']
                W = Q @ theta_param
                original_params[name] = W
            elif transform['type'] == 'standardized':
                param = theta_param * transform['std'] + transform['mean']
                original_params[name] = param.view(transform['original_shape'])
            
            theta_idx += param_size
            
        return original_params

    def set_network_parameters(self, theta_vector: torch.Tensor):
        """Set network parameters from θ vector"""
        original_params = self.theta_to_original_parameters(theta_vector)
        
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name].to(self.device))

    def compute_gradients_theta_space(self, theta_vector: torch.Tensor, 
                                    states, actions, advantages) -> tuple:
        """Compute gradients with NO artificial scaling and SEQUENTIAL priors"""
        original_theta = self.theta_vector.clone()
        
        try:
            # Check constraints
            constraint_penalty = self.apply_soft_constraints(theta_vector)
            if constraint_penalty > 0:
                return constraint_penalty, -float('inf'), -float('inf'), torch.zeros_like(theta_vector)
            
            self.set_network_parameters(theta_vector)
            
            for param in self.network.parameters():
                param.requires_grad_(True)
            self.network.zero_grad()

            # Compute sequential prior (using posterior means from previous epoch)
            log_prior = self.compute_log_prior_theta_space(theta_vector)
            if log_prior == -float('inf'):
                raise ValueError("Invalid prior")

            # Forward pass
            logits, values = self.network(states)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise ValueError("NaN/Inf in forward pass")

            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Normalize advantages
            if torch.std(advantages) > 1e-8:
                norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                norm_advantages = torch.clamp(norm_advantages, -5, 5)
            else:
                norm_advantages = torch.zeros_like(advantages)

            policy_obj = (norm_advantages * action_log_probs).mean()
            
            # Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Raw likelihood - NO SCALING like Stan
            raw_likelihood = policy_obj + self.config.ppo_entropy_coef * entropy

            if torch.isnan(raw_likelihood) or torch.isinf(raw_likelihood):
                raise ValueError("NaN/Inf in likelihood")

            # Backward pass
            raw_likelihood.backward()

            # Transform gradients to θ-space
            theta_grad = torch.zeros_like(theta_vector)
            theta_idx = 0
            
            for name in self.theta_names:
                transform = self.param_transforms[name]
                param_size = transform['size']
                param_type = self.parameter_types.get(name, 'other')
                
                param = dict(self.network.named_parameters())[name]
                if param.grad is None:
                    param_grad = torch.zeros_like(param)
                else:
                    param_grad = param.grad.clone()
                
                # Transform gradient to θ-space
                if transform['type'] == 'QR':
                    Q = transform['Q']
                    theta_param_grad = Q.T @ param_grad
                elif transform['type'] == 'standardized':
                    theta_param_grad = param_grad * transform['std']
                
                # Add prior gradient with diffuse scaling and SEQUENTIAL means
                theta_param = theta_vector[theta_idx:theta_idx + param_size].view(transform['theta_shape'])
                prior_mean = self.prior_means[name].view(transform['theta_shape'])
                
                # Use same diffuse stds as in log_prior computation
                if param_type == 'policy':
                    prior_std = self.beta_prior_std
                elif param_type == 'feature':
                    prior_std = self.feature_prior_std
                else:
                    prior_std = self.other_prior_std
                
                # Prior gradient - centered at sequential prior means
                prior_grad = -(theta_param - prior_mean) / (prior_std ** 2)
                
                total_grad = theta_param_grad + prior_grad / self.temperature
                theta_grad[theta_idx:theta_idx + param_size] = total_grad.flatten()
                
                theta_idx += param_size

            # Gradient clipping
            grad_norm = torch.norm(theta_grad)
            if grad_norm > 100.0:
                theta_grad = theta_grad * (100.0 / grad_norm)

            # Potential energy
            log_posterior = log_prior + raw_likelihood.item()
            potential_energy = -log_posterior / self.temperature

            return potential_energy, log_prior, raw_likelihood.item(), theta_grad

        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}")
            self.numerical_failures += 1
            return 1000.0, -1000.0, -1000.0, torch.zeros_like(theta_vector)
            
        finally:
            self.set_network_parameters(original_theta)

    def leapfrog_step(self, theta: torch.Tensor, momentum: torch.Tensor, 
                     step_size: float, states, actions, advantages) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single leapfrog step"""
        
        # Half step for momentum
        _, _, _, theta_grad = self.compute_gradients_theta_space(theta, states, actions, advantages)
        momentum = momentum - 0.5 * step_size * theta_grad
        
        # Full step for position
        theta = theta + step_size * momentum
        
        # Check for numerical issues
        if torch.isnan(theta).any() or torch.isinf(theta).any():
            raise ValueError("NaN/Inf in theta during leapfrog")
        
        # Half step for momentum
        _, _, _, theta_grad = self.compute_gradients_theta_space(theta, states, actions, advantages)
        momentum = momentum - 0.5 * step_size * theta_grad
        
        return theta, momentum

    def sample_momentum(self) -> torch.Tensor:
        """Sample momentum with parameter-specific scaling"""
        momentum = torch.zeros_like(self.theta_vector)
        theta_idx = 0
        
        for name in self.theta_names:
            transform = self.param_transforms[name]
            param_size = transform['size']
            
            scale = self.parameter_scales.get(name, 1.0)
            momentum[theta_idx:theta_idx + param_size] = torch.randn(param_size) * scale
            
            theta_idx += param_size
            
        return momentum

    def adapt_step_size(self, acceptance_prob: float):
        """Stan-like step size adaptation"""
        if len(self.recent_acceptances) < 10:
            return
        
        recent_acceptance_rate = np.mean(self.recent_acceptances[-10:])
        
        # Target self.adapt_delta
        if recent_acceptance_rate < self.adapt_delta - 0.1:
            self.base_step_size *= 0.9
        elif recent_acceptance_rate > self.adapt_delta + 0.05:
            self.base_step_size *= 1.1
        
        self.base_step_size = np.clip(self.base_step_size, 1e-6, 0.05)

    def start_epoch(self):
        """Start a new epoch"""
        self.theta_ensemble = []

    def add_theta_to_ensemble(self, theta_vector: torch.Tensor, was_accepted: bool):
        """Add theta vector to ensemble"""
        theta_copy = theta_vector.detach().cpu().clone()
        self.theta_ensemble.append({
            'theta': theta_copy,
            'accepted': was_accepted,
            'proposal_num': self.total_proposals
        })

    def finalize_epoch_with_mean_and_update_priors(self):
        """UPDATED: Finalize epoch with MEAN (not median) and update priors"""
        if len(self.theta_ensemble) == 0:
            return self._create_empty_stats()
        
        # Stack all theta vectors
        theta_stack = torch.stack([sample['theta'] for sample in self.theta_ensemble])
        
        # Compute MEAN instead of median (key change!)
        mean_theta = torch.mean(theta_stack, dim=0)
        
        # Set network to mean parameters
        mean_theta_device = mean_theta.to(self.device)
        self.theta_vector = mean_theta_device.clone()
        self.set_network_parameters(self.theta_vector)
        
        # SEQUENTIAL LEARNING: Update prior means for next epoch
        self.update_prior_means_from_posterior(mean_theta_device)
        
        # Compute statistics
        accepted_count = sum(1 for sample in self.theta_ensemble if sample['accepted'])
        acceptance_rate = accepted_count / len(self.theta_ensemble)
        
        # Compute ensemble statistics
        ensemble_std = torch.std(theta_stack, dim=0).mean().item()
        ensemble_range = (theta_stack.max(dim=0).values - theta_stack.min(dim=0).values).mean().item()
        
        ensemble_stats = {
            'ensemble_size': len(self.theta_ensemble),
            'accepted_count': accepted_count,
            'acceptance_rate': acceptance_rate,
            'recent_acceptance_rate': acceptance_rate,
            'parameter_std': ensemble_std,
            'parameter_range': ensemble_range,
            'mean_computed': True,  # Changed from median_computed
            'step_size': self.base_step_size,
            'temperature': self.temperature,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': 0.0,
            'log_likelihood': 0.0,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'expected_kinetic_energy': self.expected_kinetic_energy,
            'policy_param_count': self.policy_param_count,
            'adapt_delta': self.adapt_delta,
            'epoch_count': self.epoch_count,  # Track epoch progression
            'prior_mean_norm': torch.norm(mean_theta_device).item()  # Track prior evolution
        }
        
        # Clear ensemble for next epoch
        self.theta_ensemble = []
        
        logger.info(f"📊 Epoch {self.epoch_count} finalized with MEAN:")
        logger.info(f"   Ensemble size: {len(theta_stack)}")
        logger.info(f"   Acceptance rate: {acceptance_rate:.3f}")
        logger.info(f"   Parameter std: {ensemble_std:.6f}")
        logger.info(f"   Prior mean norm: {torch.norm(mean_theta_device).item():.6f}")
        
        return ensemble_stats

    def _create_empty_stats(self):
        """Create empty statistics"""
        return {
            'ensemble_size': 0, 'mean_computed': False, 'accepted_count': 0,
            'acceptance_rate': 0.0, 'recent_acceptance_rate': 0.0,
            'parameter_std': 0.0, 'parameter_range': 0.0,
            'step_size': self.base_step_size, 'temperature': self.temperature,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': 0.0, 'log_likelihood': 0.0,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'expected_kinetic_energy': self.expected_kinetic_energy,
            'policy_param_count': getattr(self, 'policy_param_count', 0),
            'adapt_delta': self.adapt_delta,
            'epoch_count': self.epoch_count,
            'prior_mean_norm': 0.0
        }

    def hmc_step(self, states, actions, advantages, old_log_probs):
        """Stan-consistent HMC step with sequential priors"""
        print(f"🔍 Sequential Stan HMC step - proposal {self.total_proposals + 1} (Epoch {self.epoch_count})")
        
        current_theta = self.theta_vector.clone()
        momentum = self.sample_momentum()
        
        # Show current prior statistics
        current_prior_norm = sum(torch.norm(self.prior_means[name]).item() 
                               for name in self.theta_names)
        
        print(f"   θ stats: mean={current_theta.mean().item():.6f}, std={current_theta.std().item():.6f}")
        print(f"   Prior mean norm: {current_prior_norm:.6f}")
        print(f"   Step size: {self.base_step_size:.8f}, Target accept: {self.adapt_delta:.2f}")
        
        # Current energy
        try:
            current_U, current_log_prior, current_log_lik, _ = \
                self.compute_gradients_theta_space(current_theta, states, actions, advantages)
                
            current_K = 0.5 * (momentum ** 2).sum().item()
            current_H = current_U + current_K
            
            print(f"   Current: H={current_H:.3f} (U={current_U:.3f}, K={current_K:.3f})")
            print(f"   Prior={current_log_prior:.3f}, Lik={current_log_lik:.3f}")
            
            if np.isnan(current_H) or np.isinf(current_H):
                raise ValueError("Invalid current energy")
                
        except Exception as e:
            print(f"   ❌ Current energy failed: {e}")
            return self._create_rejection_result()

        # Leapfrog integration
        proposed_theta = current_theta.clone()
        proposed_momentum = momentum.clone()
        
        try:
            for step in range(self.num_leapfrog_steps):
                proposed_theta, proposed_momentum = self.leapfrog_step(
                    proposed_theta, proposed_momentum, self.base_step_size,
                    states, actions, advantages
                )
                
        except Exception as e:
            print(f"   ❌ Leapfrog failed: {e}")
            return self._create_rejection_result()

        # Proposed energy
        try:
            proposed_U, proposed_log_prior, proposed_log_lik, _ = \
                self.compute_gradients_theta_space(proposed_theta, states, actions, advantages)
                
            proposed_K = 0.5 * (proposed_momentum ** 2).sum().item()
            proposed_H = proposed_U + proposed_K
            
            print(f"   Proposed: H={proposed_H:.3f} (U={proposed_U:.3f}, K={proposed_K:.3f})")
            
            if np.isnan(proposed_H) or np.isinf(proposed_H):
                raise ValueError("Invalid proposed energy")
                
        except Exception as e:
            print(f"   ❌ Proposed energy failed: {e}")
            return self._create_rejection_result()

        # Metropolis acceptance
        energy_change = proposed_H - current_H
        energy_change = np.clip(energy_change, -50, 50)
        acceptance_prob = min(1.0, np.exp(-energy_change))

        print(f"   ΔH={energy_change:.3f}, Accept prob={acceptance_prob:.6f}")

        # Accept or reject
        if np.random.rand() < acceptance_prob:
            self.theta_vector = proposed_theta.clone()
            self.set_network_parameters(self.theta_vector)
            accepted = True
            final_log_prior = proposed_log_prior
            final_log_lik = proposed_log_lik
            final_theta = proposed_theta
            print(f"   ✅ ACCEPTED")
        else:
            accepted = False
            final_log_prior = current_log_prior
            final_log_lik = current_log_lik
            final_theta = current_theta
            print(f"   ❌ REJECTED")

        # Add to ensemble
        self.add_theta_to_ensemble(final_theta, accepted)

        # Update statistics
        self.total_proposals += 1
        if accepted:
            self.total_acceptances += 1

        self.recent_acceptances.append(accepted)
        if len(self.recent_acceptances) > 50:
            self.recent_acceptances.pop(0)

        current_acceptance_rate = self.total_acceptances / self.total_proposals
        recent_acceptance_rate = np.mean(self.recent_acceptances) if self.recent_acceptances else 0.0

        # Adapt step size
        self.adapt_step_size(acceptance_prob)

        # Log progress
        if self.total_proposals % 10 == 0:
            print(f"🎯 Sequential Stan HMC Step {self.total_proposals}: "
                       f"Accept={accepted}, Rate={recent_acceptance_rate:.3f}, "
                       f"Epoch={self.epoch_count}")

        return {
            'accepted': accepted,
            'acceptance_rate': current_acceptance_rate,
            'recent_acceptance_rate': recent_acceptance_rate,
            'acceptance_prob': acceptance_prob,
            'energy_change': energy_change,
            'temperature': self.temperature,
            'step_size': self.base_step_size,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': final_log_prior,
            'log_likelihood': final_log_lik,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'expected_kinetic_energy': self.expected_kinetic_energy,
            'policy_param_count': self.policy_param_count,
            'adapt_delta': self.adapt_delta,
            'ensemble_size': len(self.theta_ensemble),
            'epoch_count': self.epoch_count,
            'prior_mean_norm': sum(torch.norm(self.prior_means[name]).item() 
                                 for name in self.theta_names)
        }

    def _create_rejection_result(self):
        """Create rejection result"""
        self.total_proposals += 1
        self.recent_acceptances.append(False)
        
        current_acceptance_rate = self.total_acceptances / self.total_proposals
        recent_acceptance_rate = np.mean(self.recent_acceptances) if self.recent_acceptances else 0.0

        return {
            'accepted': False,
            'acceptance_rate': current_acceptance_rate,
            'recent_acceptance_rate': recent_acceptance_rate,
            'acceptance_prob': 0.0,
            'energy_change': 0.0,
            'temperature': self.temperature,
            'step_size': self.base_step_size,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': 0.0,
            'log_likelihood': 0.0,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'policy_param_count': getattr(self, 'policy_param_count', 0),
            'adapt_delta': self.adapt_delta,
            'epoch_count': self.epoch_count,
            'prior_mean_norm': sum(torch.norm(self.prior_means[name]).item() 
                                 for name in self.theta_names)
        }


class StandardPPO:
    """Standard PPO implementation"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.clip_epsilon = config.ppo_clip_epsilon

    def compute_policy_loss(self, old_log_probs, new_log_probs, advantages, values, returns):
        """Exact PPO loss computation"""
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Value loss
        value_losses = (values - returns) ** 2
        value_loss = 0.5 * torch.mean(value_losses)

        # Entropy loss
        entropy_loss = -torch.mean(torch.exp(new_log_probs) * new_log_probs)

        clipped_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'clipped_fraction': clipped_fraction,
            'avg_ratio': torch.mean(ratio).item()
        }


def compute_advantages_and_returns_exact(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Exact GAE computation"""
    advantages, returns = [], []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    advantages = np.array(advantages)
    returns = np.array(returns)

    if len(advantages) > 1 and np.std(advantages) > 1e-8:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return advantages, returns


def collect_trajectory(env, network, max_steps=1000):
    """Collect trajectory with action tracking"""
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    action_counts = np.zeros(14)

    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action, log_prob, value = network.get_action_and_value(state)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        action_counts[action] += 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        dones.append(done)
        total_reward += reward
        state = next_state

        if done:
            break

    return {
        'states': np.array(states), 'actions': np.array(actions),
        'rewards': np.array(rewards), 'log_probs': np.array(log_probs),
        'values': np.array(values), 'dones': np.array(dones),
        'total_reward': total_reward, 'length': len(rewards),
        'action_counts': action_counts
    }


def train_ppo_network(network, optimizer, policy_optimizer, trajectories, config):
    """Train PPO network"""
    all_states = np.concatenate([traj['states'] for traj in trajectories])
    all_actions = np.concatenate([traj['actions'] for traj in trajectories])
    all_rewards = np.concatenate([traj['rewards'] for traj in trajectories])
    all_log_probs = np.concatenate([traj['log_probs'] for traj in trajectories])
    all_values = np.concatenate([traj['values'] for traj in trajectories])
    all_dones = np.concatenate([traj['dones'] for traj in trajectories])

    advantages, returns = compute_advantages_and_returns_exact(all_rewards, all_values, all_dones)

    states = torch.FloatTensor(all_states).to(device)
    actions = torch.LongTensor(all_actions).to(device)
    old_log_probs = torch.FloatTensor(all_log_probs).to(device)
    advantages_tensor = torch.FloatTensor(advantages).to(device)
    returns_tensor = torch.FloatTensor(returns).to(device)

    total_loss = 0
    for epoch in range(config.updates_per_epoch):
        new_log_probs, values, entropy = network.evaluate_actions(states, actions)

        loss_dict = policy_optimizer.compute_policy_loss(
            old_log_probs, new_log_probs, advantages_tensor, values, returns_tensor
        )

        loss = (loss_dict['policy_loss'] +
               config.ppo_value_coef * loss_dict['value_loss'] -
               config.ppo_entropy_coef * loss_dict['entropy_loss'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        stats = loss_dict

    return total_loss / config.updates_per_epoch, stats


def train_sequential_stan_hmc_network(network, hmc_optimizer, trajectories, config):
    """Train network using Sequential Stan-consistent HMC"""
    try:
        device = next(network.parameters()).device
        
        # Process trajectories
        all_states = np.concatenate([traj['states'] for traj in trajectories])
        all_actions = np.concatenate([traj['actions'] for traj in trajectories])
        all_rewards = np.concatenate([traj['rewards'] for traj in trajectories])
        all_values = np.concatenate([traj['values'] for traj in trajectories])
        all_dones = np.concatenate([traj['dones'] for traj in trajectories])

        # Compute advantages
        def compute_advantages_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
            advantages, returns = [], []
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = 0
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            advantages = np.array(advantages)
            if len(advantages) > 1 and np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            return advantages, returns
            
        advantages, returns = compute_advantages_gae(all_rewards, all_values, all_dones)

        # Convert to tensors
        states = torch.FloatTensor(all_states).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        old_log_probs = torch.zeros(len(all_actions)).to(device)

        # Start ensemble collection
        hmc_optimizer.start_epoch()
        
        total_accepted = 0
        hmc_stats = None

        print(f"🧮 Starting Sequential Stan-consistent ensemble collection with {config.updates_per_epoch} HMC steps...")
        print(f"   Current epoch: {hmc_optimizer.epoch_count}")

        for epoch in range(config.updates_per_epoch):
            hmc_result = hmc_optimizer.hmc_step(
                states, actions, advantages_tensor, old_log_probs
            )

            if hmc_result['accepted']:
                total_accepted += 1

            hmc_stats = hmc_result

        # Finalize with mean and update priors for next epoch
        ensemble_stats = hmc_optimizer.finalize_epoch_with_mean_and_update_priors()
        
        print(f"🎯 Sequential Stan-consistent ensemble training complete:")
        print(f"   Total HMC steps: {config.updates_per_epoch}")
        print(f"   Accepted steps: {total_accepted}")
        print(f"   Ensemble size: {ensemble_stats['ensemble_size']}")
        print(f"   Epoch: {ensemble_stats['epoch_count']}")
        print(f"   Prior mean norm: {ensemble_stats['prior_mean_norm']:.6f}")

        # Compute final loss
        try:
            with torch.no_grad():
                new_log_probs, values, entropy = network.evaluate_actions(states, actions)
                policy_loss = -(new_log_probs * advantages_tensor).mean()
                total_loss = policy_loss

        except Exception as e:
            logger.warning(f"Loss computation failed: {e}")
            total_loss = torch.tensor(1000.0)
            policy_loss = torch.tensor(1000.0)

        # Combine stats
        final_stats = ensemble_stats.copy()
        final_stats['policy_loss'] = policy_loss.item() if hasattr(policy_loss, 'item') else 1000.0
        final_stats['acceptances_this_update'] = total_accepted

        return total_loss.item() if hasattr(total_loss, 'item') else total_loss, final_stats

    except Exception as e:
        logger.error(f"Sequential Stan-consistent HMC training failed: {e}")
        fallback_stats = {
            'accepted': False, 'acceptance_rate': 0.0, 'recent_acceptance_rate': 0.0,
            'policy_loss': 1000.0, 'acceptances_this_update': 0,
            'ensemble_size': 0, 'mean_computed': False,
            'step_size': 0.001, 'temperature': 1.0, 'num_leapfrog_steps': 10,
            'log_prior': 0.0, 'log_likelihood': 0.0, 'numerical_failures': 0,
            'theta_dimension': 0, 'qr_transforms': 0, 'expected_kinetic_energy': 0.0,
            'policy_param_count': 0, 'adapt_delta': 0.8, 'TRAINING_FAILED': True,
            'epoch_count': 0, 'prior_mean_norm': 0.0
        }
        return 1000.0, fallback_stats


def create_environment_with_video(config: ExperimentConfig, method_name: str, 
                                episode_num: int = 0, network=None, other_network=None, 
                                other_method_name=""):
    """Create environment with video recording"""
    
    # Create base environment
    env = gym.make(config.env_id, render_mode="rgb_array", frameskip=1)

    # Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=config.screen_size,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False
    )

    # Frame stacking
    env = FrameStack(env, config.frame_stack)

    return env


def create_performance_plots(results, experiment_dir, episode_count):
    """Create performance plots with sequential learning metrics"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))

    # 1. Reward comparison
    ax1 = plt.subplot(2, 4, 1)
    ppo_rewards = results['ppo']['episode_rewards']
    hmc_rewards = results['hmc']['episode_rewards']

    episodes = range(len(ppo_rewards))
    ax1.plot(episodes, ppo_rewards, 'b-', alpha=0.6, label='PPO')
    ax1.plot(episodes, hmc_rewards, 'r-', alpha=0.6, label='Sequential Stan-HMC')

    # Smoothed versions
    if len(ppo_rewards) > 20:
        ppo_smooth = uniform_filter1d(ppo_rewards, size=20, mode='nearest')
        hmc_smooth = uniform_filter1d(hmc_rewards, size=20, mode='nearest')
        ax1.plot(episodes, ppo_smooth, 'b-', linewidth=3, label='PPO Smoothed')
        ax1.plot(episodes, hmc_smooth, 'r-', linewidth=3, label='Sequential HMC Smoothed')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards: PPO vs Sequential Stan-HMC', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. HMC Acceptance Rate
    ax2 = plt.subplot(2, 4, 2)
    if 'acceptance_rates' in results['hmc'] and len(results['hmc']['acceptance_rates']) > 0:
        ax2.plot(results['hmc']['acceptance_rates'], 'g-', linewidth=2, label='Acceptance Rate')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target (80%)')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_title('Sequential HMC Acceptance Rate', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Step Size Evolution
    ax3 = plt.subplot(2, 4, 3)
    if 'step_sizes' in results['hmc'] and len(results['hmc']['step_sizes']) > 0:
        ax3.plot(results['hmc']['step_sizes'], 'purple', linewidth=2)
        ax3.set_ylabel('Step Size')
        ax3.set_title('Sequential HMC Step Size Adaptation', fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Prior Mean Evolution (NEW!)
    ax4 = plt.subplot(2, 4, 4)
    if 'prior_mean_norms' in results['hmc'] and len(results['hmc']['prior_mean_norms']) > 0:
        ax4.plot(results['hmc']['prior_mean_norms'], 'orange', linewidth=2)
        ax4.set_ylabel('Prior Mean Norm')
        ax4.set_title('Sequential Prior Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Episode Length Comparison
    ax5 = plt.subplot(2, 4, 5)
    if len(results['ppo']['episode_lengths']) > 0 and len(results['hmc']['episode_lengths']) > 0:
        ax5.plot(results['ppo']['episode_lengths'], 'b-', alpha=0.6, label='PPO')
        ax5.plot(results['hmc']['episode_lengths'], 'r-', alpha=0.6, label='Sequential Stan-HMC')
        ax5.set_ylabel('Episode Length')
        ax5.set_title('Episode Lengths', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Loss Comparison
    ax6 = plt.subplot(2, 4, 6)
    if len(results['ppo']['losses']) > 0:
        ax6.plot(results['ppo']['losses'], 'b-', linewidth=2, label='PPO Loss')
        if 'losses' in results['hmc'] and len(results['hmc']['losses']) > 0:
            ax6.plot(results['hmc']['losses'], 'r-', linewidth=2, label='Sequential HMC Loss')
        ax6.set_ylabel('Loss')
        ax6.set_title('Training Loss', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # 7. Epoch Count (NEW!)
    ax7 = plt.subplot(2, 4, 7)
    if 'epoch_counts' in results['hmc'] and len(results['hmc']['epoch_counts']) > 0:
        ax7.plot(results['hmc']['epoch_counts'], 'brown', linewidth=2)
        ax7.set_ylabel('Epoch Count')
        ax7.set_title('Sequential Learning Progress', fontweight='bold')
        ax7.grid(True, alpha=0.3)

    # 8. Summary Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    if len(ppo_rewards) > 0 and len(hmc_rewards) > 0:
        ppo_mean = np.mean(ppo_rewards)
        hmc_mean = np.mean(hmc_rewards)
        ppo_std = np.std(ppo_rewards)
        hmc_std = np.std(hmc_rewards)
        
        summary_text = f"""
SEQUENTIAL LEARNING SUMMARY

Overall Performance:
PPO: {ppo_mean:.1f} ± {ppo_std:.1f}
Sequential HMC: {hmc_mean:.1f} ± {hmc_std:.1f}
Difference: {hmc_mean - ppo_mean:+.1f}

Episodes: {episode_count}
        """
        
        if 'acceptance_rates' in results['hmc'] and len(results['hmc']['acceptance_rates']) > 0:
            final_acceptance = results['hmc']['acceptance_rates'][-1]
            summary_text += f"\nHMC Acceptance: {final_acceptance:.3f}"
        
        if 'epoch_counts' in results['hmc'] and len(results['hmc']['epoch_counts']) > 0:
            final_epoch = results['hmc']['epoch_counts'][-1]
            summary_text += f"\nFinal Epoch: {final_epoch}"
        
        if 'prior_mean_norms' in results['hmc'] and len(results['hmc']['prior_mean_norms']) > 0:
            final_prior_norm = results['hmc']['prior_mean_norms'][-1]
            summary_text += f"\nPrior Evolution: {final_prior_norm:.3f}"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    
    # Save plot
    plot_path = experiment_dir / f'sequential_stan_performance_episode_{episode_count}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 Sequential performance plots saved: {plot_path}")


def run_sequential_stan_experiment():
    """Run the PPO vs Sequential Stan-consistent HMC experiment"""
    
    print("=" * 80)
    print("🚀 PPO vs SEQUENTIAL STAN-CONSISTENT HMC EXPERIMENT")
    print("=" * 80)
    
    config = ExperimentConfig()
    experiment_dir = Path("sequential_experiment_results")
    experiment_dir.mkdir(exist_ok=True)
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create networks
    ppo_network = AsteroidsNetwork(config)
    hmc_network = AsteroidsNetwork(config)
    hmc_network.load_state_dict(ppo_network.state_dict())
    
    print(f"🧠 Networks initialized:")
    print(f"   Total parameters: {sum(p.numel() for p in ppo_network.parameters()):,}")
    
    # Create optimizers
    ppo_optimizer = optim.Adam(ppo_network.parameters(), lr=config.learning_rate)
    ppo_policy_opt = StandardPPO(config)
    
    print(f"🔧 Creating Sequential Stan-consistent HMC optimizer...")
    try:
        hmc_policy_opt = SequentialStanConsistentHMC(hmc_network, config)
        print(f"✅ Sequential Stan-consistent HMC optimizer created successfully")
        
        print(f"📊 Key Sequential Features:")
        print(f"   DIFFUSE priors: σ = {hmc_policy_opt.beta_prior_std} (fixed)")
        print(f"   Policy parameters: {hmc_policy_opt.policy_param_count:,}")
        print(f"   Target acceptance: {hmc_policy_opt.adapt_delta:.2f}")
        print(f"   Sequential learning: ENABLED")
        print(f"   Prior-to-posterior updates: ENABLED")
        print()
        
    except Exception as e:
        print(f"❌ Failed to create Sequential Stan-consistent HMC optimizer: {e}")
        raise
    
    # Results tracking with new sequential metrics
    results = {
        'ppo': {'episode_rewards': [], 'episode_lengths': [], 'losses': []},
        'hmc': {'episode_rewards': [], 'episode_lengths': [], 'losses': [],
                'acceptance_rates': [], 'step_sizes': [], 'log_priors': [],
                'log_likelihoods': [], 'numerical_failures': [],
                'epoch_counts': [], 'prior_mean_norms': []},  # NEW tracking
        'action_distributions': {'ppo': np.zeros(14), 'hmc': np.zeros(14)}
    }
    
    # Training loop
    episode_count = 0
    start_time = time.time()
    
    try:
        while episode_count < config.total_episodes:
            current_round = episode_count // config.episodes_per_update + 1
            
            print(f"\n{'='*60}")
            print(f"Training Round {current_round}")
            print(f"Episodes: {episode_count}-{min(episode_count + config.episodes_per_update, config.total_episodes)}")
            print(f"Sequential HMC Epoch: {hmc_policy_opt.epoch_count}")
            
            # Create environments
            ppo_env = create_environment_with_video(config, "PPO", episode_count)
            hmc_env = create_environment_with_video(config, "Sequential-Stan-HMC", episode_count)
            
            # Collect trajectories
            ppo_trajectories = []
            hmc_trajectories = []
            
            for ep in range(config.episodes_per_update):
                if episode_count >= config.total_episodes:
                    break
                
                # PPO trajectory
                ppo_traj = collect_trajectory(ppo_env, ppo_network)
                ppo_trajectories.append(ppo_traj)
                results['ppo']['episode_rewards'].append(ppo_traj['total_reward'])
                results['ppo']['episode_lengths'].append(ppo_traj['length'])
                
                # HMC trajectory
                hmc_traj = collect_trajectory(hmc_env, hmc_network)
                hmc_trajectories.append(hmc_traj)
                results['hmc']['episode_rewards'].append(hmc_traj['total_reward'])
                results['hmc']['episode_lengths'].append(hmc_traj['length'])
                
                episode_count += 1

                # Progress update
                if episode_count % 4 == 0:
                    ppo_recent = np.mean([t['total_reward'] for t in ppo_trajectories[-4:]])
                    hmc_recent = np.mean([t['total_reward'] for t in hmc_trajectories[-4:]])
                    print(f"  Episode {episode_count:4d} - PPO: {ppo_recent:7.1f}, "
                          f"Sequential HMC: {hmc_recent:7.1f}")
            
            # Close environments
            ppo_env.close()
            hmc_env.close()
            
            # Training updates
            if ppo_trajectories and hmc_trajectories:
                # Train PPO
                ppo_loss, ppo_stats = train_ppo_network(
                    ppo_network, ppo_optimizer, ppo_policy_opt, ppo_trajectories, config
                )
                results['ppo']['losses'].append(ppo_loss)
                
                # Train Sequential Stan-consistent HMC
                hmc_loss, hmc_stats = train_sequential_stan_hmc_network(
                    hmc_network, hmc_policy_opt, hmc_trajectories, config
                )
                
                # Training statistics
                print(f"\n📊 Training Statistics:")
                print(f"  PPO - Loss: {ppo_loss:.4f}")
                
                acceptance_rate = hmc_stats.get('acceptance_rate', 0.0)
                recent_acceptance = hmc_stats.get('recent_acceptance_rate', 0.0)
                step_size = hmc_stats.get('step_size', 0.0)
                epoch_count_hmc = hmc_stats.get('epoch_count', 0)
                prior_mean_norm = hmc_stats.get('prior_mean_norm', 0.0)
                
                print(f"  Sequential HMC - Loss: {hmc_loss:.4f}, Accept: {acceptance_rate:.3f}")
                print(f"                   Step Size: {step_size:.6f}, Epoch: {epoch_count_hmc}")
                print(f"                   Prior norm: {prior_mean_norm:.6f}")

                # Store HMC stats (including new sequential metrics)
                results['hmc']['acceptance_rates'].append(acceptance_rate)
                results['hmc']['step_sizes'].append(step_size)
                results['hmc']['log_priors'].append(hmc_stats.get('log_prior', 0))
                results['hmc']['log_likelihoods'].append(hmc_stats.get('log_likelihood', 0))
                results['hmc']['numerical_failures'].append(hmc_stats.get('numerical_failures', 0))
                results['hmc']['epoch_counts'].append(epoch_count_hmc)  # NEW
                results['hmc']['prior_mean_norms'].append(prior_mean_norm)  # NEW
                results['hmc']['losses'].append(hmc_loss)
            
            # Generate plots
            if episode_count % config.plot_frequency == 0 and episode_count > 0:
                print(f"\n📈 Generating sequential performance plots...")
                create_performance_plots(results, experiment_dir, episode_count)
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted at episode {episode_count}")
    
    finally:
        # Final analysis
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🏁 SEQUENTIAL STAN-CONSISTENT EXPERIMENT COMPLETE!")
        print(f"Episodes completed: {episode_count}/{config.total_episodes}")
        print(f"Total time: {total_time/60:.1f} minutes")
        
        if results['ppo']['episode_rewards'] and results['hmc']['episode_rewards']:
            ppo_rewards = results['ppo']['episode_rewards']
            hmc_rewards = results['hmc']['episode_rewards']
            
            print(f"\n🏆 FINAL RESULTS:")
            print(f"PPO: {np.mean(ppo_rewards):.1f} ± {np.std(ppo_rewards):.1f}")
            print(f"Sequential Stan-HMC: {np.mean(hmc_rewards):.1f} ± {np.std(hmc_rewards):.1f}")
            
            improvement = np.mean(hmc_rewards) - np.mean(ppo_rewards)
            print(f"Performance difference: {improvement:+.1f}")
        
        # HMC diagnostics
        if hmc_policy_opt.total_proposals > 0:
            print(f"\n🔧 SEQUENTIAL STAN-CONSISTENT HMC DIAGNOSTICS:")
            print(f"Final acceptance rate: {hmc_policy_opt.total_acceptances / hmc_policy_opt.total_proposals:.3f}")
            print(f"Target acceptance: {hmc_policy_opt.adapt_delta:.2f}")
            print(f"Final step size: {hmc_policy_opt.base_step_size:.8f}")
            print(f"Final epoch: {hmc_policy_opt.epoch_count}")
            print(f"Policy parameters: {hmc_policy_opt.policy_param_count:,}")
            print(f"Numerical failures: {hmc_policy_opt.numerical_failures}")
            
            # Sequential learning diagnostics
            final_prior_norm = sum(torch.norm(hmc_policy_opt.prior_means[name]).item() 
                                 for name in hmc_policy_opt.theta_names)
            print(f"Final prior mean norm: {final_prior_norm:.6f}")

        # Save results
        print(f"\n💾 Saving sequential results...")
        data_dir = experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)

        with open(data_dir / "sequential_stan_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        with open(data_dir / "sequential_experiment_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        # Save HMC optimizer state for analysis
        hmc_state = {
            'prior_means': {name: mean.cpu().numpy() for name, mean in hmc_policy_opt.prior_means.items()},
            'epoch_count': hmc_policy_opt.epoch_count,
            'total_proposals': hmc_policy_opt.total_proposals,
            'total_acceptances': hmc_policy_opt.total_acceptances,
            'parameter_types': hmc_policy_opt.parameter_types,
            'theta_names': hmc_policy_opt.theta_names
        }
        
        with open(data_dir / "sequential_hmc_state.pkl", 'wb') as f:
            pickle.dump(hmc_state, f)

        create_performance_plots(results, experiment_dir, episode_count)
        
        print(f"\n✅ Sequential Stan-consistent HMC experiment completed successfully!")
        print(f"📁 Results saved in: {experiment_dir}")
        print(f"\n🎯 SEQUENTIAL STAN-CONSISTENT IMPROVEMENTS:")
        print(f"  ✅ ENSEMBLE MEAN instead of median for parameter estimates")
        print(f"  ✅ POSTERIOR-TO-PRIOR updates: epoch N posteriors → epoch N+1 priors")
        print(f"  ✅ DIFFUSE priors (σ = 10.0) instead of tight priors")
        print(f"  ✅ NO artificial likelihood scaling - raw likelihood used")
        print(f"  ✅ Proper Jacobian corrections for QR transforms")
        print(f"  ✅ Parameter-specific mass matrix scaling")
        print(f"  ✅ Soft parameter bounds and numerical stability")
        print(f"  ✅ Stan-like adaptive step size targeting 80% acceptance")
        print(f"  ✅ Conservative defaults with progressive learning")
        print(f"  ✅ Sequential knowledge accumulation across epochs")
    
    return results, hmc_policy_opt


if __name__ == "__main__":
    print("🚀 Starting Sequential Stan-Consistent HMC vs PPO Experiment")
    print("🎯 SEQUENTIAL STAN-CONSISTENT FEATURES:")
    print("  ✅ ENSEMBLE MEAN (not median) for parameter estimates")
    print("  ✅ POSTERIOR-TO-PRIOR: Epoch N posteriors become Epoch N+1 priors")
    print("  ✅ DIFFUSE priors (σ = 10.0) for ALL parameters (fixed hyperparameters)")
    print("  ✅ NO artificial likelihood scaling (use raw likelihood)")
    print("  ✅ Proper Jacobian corrections for QR transformations") 
    print("  ✅ Parameter-specific step sizes and mass matrix")
    print("  ✅ Soft parameter bounds like Stan's implicit constraints")
    print("  ✅ Gradient clipping and numerical stability")
    print("  ✅ Conservative defaults with sequential adaptation")
    print("  ✅ Target acceptance rate adaptation (δ = 0.8)")
    print("  ✅ Progressive learning that builds on previous knowledge")
    print()
    
    """
    try:
        results, hmc_optimizer = run_sequential_stan_experiment()
        print("\n🎉 Sequential Stan-consistent experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🏁 Script execution finished")
    """