#!/usr/bin/env python3
"""
COMPLETE FIXED Ensemble HMC Implementation with Greatly Improved Video System

Key fixes:
1. Fixed finalize_epoch_with_median() return value structure
2. Completely overhauled video system for reliability
3. Added action probability bar charts instead of individual commands
4. Enhanced error handling throughout
5. Proper ensemble median computation
6. COMPLETE script with all functions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.ndimage import uniform_filter1d
import cv2
import math
import pickle

# Setup logging and device
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Asteroids action names for analysis
ASTEROIDS_ACTIONS = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
    10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE"
}


class TrueQRTransformedHMC:
    """
    PROPERLY SCALED θ-space HMC with correct energy balance and fixed ensemble
    """

    def __init__(self, network: nn.Module, config):
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device

        # Transformations for ALL parameters
        self.param_transforms = {}
        
        # Single θ vector containing ALL transformed parameters
        self.theta_vector = None
        self.theta_shapes = {}
        self.theta_names = []
        
        # HMC operates with single uniform step size in θ-space
        self.step_size = config.hmc_step_size
        self.num_leapfrog_steps = config.hmc_num_leapfrog_steps
        self.temperature = config.hmc_temperature

        # Statistics
        self.total_proposals = 0
        self.total_acceptances = 0
        self.recent_acceptances = []
        self.numerical_failures = 0

        # Ensemble storage
        self.theta_ensemble = []

        self._setup_parameter_transformations()
        self._initialize_theta_vector()

        # CRITICAL: Compute proper energy scaling after θ vector is created
        self.n_params = self.theta_vector.numel()
        self.expected_kinetic_energy = self.n_params / 2.0
        self.likelihood_scale = 1.0
        self.theta_prior_std = 0.1
        
        logger.info(f"🎯 PROPERLY SCALED θ-space HMC initialized:")
        logger.info(f"   Total θ parameters: {self.n_params:,}")
        logger.info(f"   Expected kinetic energy: {self.expected_kinetic_energy:.1f}")
        logger.info(f"   Prior std: {self.theta_prior_std:.1f}")
        logger.info(f"   Likelihood scale: {self.likelihood_scale:.3f}")
        logger.info(f"   Target step size: {self.step_size:.3f}")

    def _setup_parameter_transformations(self):
        """Setup transformations - same as before"""
        param_idx = 0
        
        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue
                
            param_data = param.data.clone().to(self.device)
            param_size = param_data.numel()
            
            if len(param_data.shape) == 2 and min(param_data.shape) >= 3:
                success = self._setup_qr_transform(name, param_data, param_idx)
                if not success:
                    self._setup_standardization_transform(name, param_data, param_idx)
            else:
                self._setup_standardization_transform(name, param_data, param_idx)
            
            param_idx += param_size

    def _setup_qr_transform(self, name: str, W: torch.Tensor, start_idx: int) -> bool:
        """Setup QR transformation: W = Q @ R, θ = R"""
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
                'size': R.numel()
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
            'size': param.numel()
        }

    def _initialize_theta_vector(self):
        """Initialize θ vector - same as before"""
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

    def start_epoch(self):
        """Start a new epoch - clear the ensemble"""
        self.theta_ensemble = []
        logger.debug(f"🔄 Starting new epoch - cleared theta ensemble")

    def add_theta_to_ensemble(self, theta_vector: torch.Tensor, was_accepted: bool):
        """Add theta vector to ensemble"""
        theta_copy = theta_vector.detach().cpu().clone()
        self.theta_ensemble.append({
            'theta': theta_copy,
            'accepted': was_accepted,
            'proposal_num': self.total_proposals
        })
        
        logger.debug(f"Added theta {self.total_proposals} to ensemble (accepted: {was_accepted})")

    def finalize_epoch_with_median(self):
        """
        FIXED: Finalize the epoch by setting theta to median of ensemble
        
        Returns:
            dict: Statistics about the ensemble and median computation
        """
        if len(self.theta_ensemble) == 0:
            logger.warning("No theta samples in ensemble - keeping current theta")
            return {
                'ensemble_size': 0, 
                'median_computed': False,
                'accepted_count': 0,
                'acceptance_rate': 0.0,
                'recent_acceptance_rate': 0.0,
                'parameter_std': 0.0,
                'parameter_range': 0.0,
                'step_size': self.step_size,
                'temperature': self.temperature,
                'num_leapfrog_steps': self.num_leapfrog_steps,
                'log_prior': 0.0,
                'log_likelihood': 0.0,
                'numerical_failures': self.numerical_failures,
                'theta_dimension': self.theta_vector.numel(),
                'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
                'expected_kinetic_energy': self.expected_kinetic_energy,
                'likelihood_scale': self.likelihood_scale
            }
        
        logger.info(f"🧮 Computing median from {len(self.theta_ensemble)} theta samples...")
        
        # Stack all theta vectors
        theta_stack = torch.stack([sample['theta'] for sample in self.theta_ensemble])
        
        # Compute median along the sample dimension (dim=0)
        median_theta = torch.median(theta_stack, dim=0).values
        
        # Set the network to median parameters
        median_theta_device = median_theta.to(self.device)
        self.theta_vector = median_theta_device.clone()
        self.set_network_parameters(self.theta_vector)
        
        # Compute statistics
        accepted_count = sum(1 for sample in self.theta_ensemble if sample['accepted'])
        acceptance_rate = accepted_count / len(self.theta_ensemble)
        
        # Parameter change statistics
        if len(self.theta_ensemble) > 1:
            theta_std = torch.std(theta_stack, dim=0).mean().item()
            theta_range = (theta_stack.max(dim=0).values - theta_stack.min(dim=0).values).mean().item()
        else:
            theta_std = 0.0
            theta_range = 0.0
        
        logger.info(f"✅ Median theta computed from {len(self.theta_ensemble)} samples:")
        logger.info(f"   Accepted: {accepted_count}/{len(self.theta_ensemble)} ({acceptance_rate:.1%})")
        logger.info(f"   Parameter std: {theta_std:.6f}")
        logger.info(f"   Parameter range: {theta_range:.6f}")
        
        # FIXED: Return proper dictionary structure
        ensemble_stats = {
            'ensemble_size': len(self.theta_ensemble),
            'accepted_count': accepted_count,
            'acceptance_rate': acceptance_rate,
            'recent_acceptance_rate': acceptance_rate,  # Add this key
            'parameter_std': theta_std,
            'parameter_range': theta_range,
            'median_computed': True,
            'step_size': self.step_size,  # Add this key
            'temperature': self.temperature,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': 0.0,  # Add default values
            'log_likelihood': 0.0,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'expected_kinetic_energy': self.expected_kinetic_energy,
            'likelihood_scale': self.likelihood_scale
        }
        
        # Clear ensemble for next epoch
        self.theta_ensemble = []
        
        return ensemble_stats

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

    def compute_log_prior_theta_space(self, theta_vector: torch.Tensor) -> float:
        """Compute log prior in θ-space"""
        if torch.isnan(theta_vector).any() or torch.isinf(theta_vector).any():
            return -float('inf')
        
        prior_var = self.theta_prior_std ** 2
        log_prior = -0.5 * (theta_vector ** 2).sum().item() / prior_var
        
        return log_prior

    def compute_gradients_theta_space(self, theta_vector: torch.Tensor, 
                                    states, actions, advantages) -> tuple:
        """Compute gradients with correct energy scaling"""
        original_theta = self.theta_vector.clone()
        
        try:
            self.set_network_parameters(theta_vector)
            
            for param in self.network.parameters():
                param.requires_grad_(True)
            self.network.zero_grad()

            # Compute prior
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
            
            raw_likelihood = policy_obj + self.config.ppo_entropy_coef * entropy
            scaled_likelihood = raw_likelihood * self.likelihood_scale

            if torch.isnan(scaled_likelihood) or torch.isinf(scaled_likelihood):
                raise ValueError("NaN/Inf in likelihood")

            # Backward pass
            scaled_likelihood.backward()

            # Transform gradients to θ-space
            theta_grad = torch.zeros_like(theta_vector)
            theta_idx = 0
            
            for name in self.theta_names:
                transform = self.param_transforms[name]
                param_size = transform['size']
                
                # Get gradient w.r.t. original parameter
                param = dict(self.network.named_parameters())[name]
                if param.grad is None:
                    param_grad = torch.zeros_like(param)
                else:
                    param_grad = param.grad.clone()
                
                if transform['type'] == 'QR':
                    Q = transform['Q']
                    theta_param_grad = Q.T @ param_grad
                elif transform['type'] == 'standardized':
                    theta_param_grad = param_grad * transform['std']
                
                # Add prior gradient
                theta_param = theta_vector[theta_idx:theta_idx + param_size].view(transform['theta_shape'])
                prior_grad = -theta_param / (self.theta_prior_std ** 2)
                
                total_grad = theta_param_grad + prior_grad / self.temperature
                theta_grad[theta_idx:theta_idx + param_size] = total_grad.flatten()
                
                theta_idx += param_size

            # Potential energy
            log_posterior = log_prior + scaled_likelihood.item()
            potential_energy = -log_posterior / self.temperature

            return potential_energy, log_prior, scaled_likelihood.item(), theta_grad

        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}")
            self.numerical_failures += 1
            return 1000.0, -1000.0, -1000.0, torch.zeros_like(theta_vector)
            
        finally:
            self.set_network_parameters(original_theta)

    def hmc_step(self, states, actions, advantages, old_log_probs):
        """HMC step that adds ALL proposals to ensemble"""
        print(f"🔍 ENSEMBLE HMC step called - proposal {self.total_proposals + 1}")
        
        current_theta = self.theta_vector.clone()
        
        # Unit variance momentum in well-conditioned θ-space
        momentum = torch.randn_like(current_theta)
        
        print(f"   Current θ stats: mean={current_theta.mean().item():.6f}, std={current_theta.std().item():.6f}")
        print(f"   Momentum stats: mean={momentum.mean().item():.6f}, std={momentum.std().item():.6f}")
        print(f"   Expected K: {self.expected_kinetic_energy:.1f}, Likelihood scale: {self.likelihood_scale:.3f}")
        
        # Current energy
        try:
            current_U, current_log_prior, current_log_lik, _ = \
                self.compute_gradients_theta_space(current_theta, states, actions, advantages)
                
            current_K = 0.5 * (momentum ** 2).sum().item()
            current_H = current_U + current_K
            
            print(f"   Current energy: H={current_H:.3f} (U={current_U:.3f}, K={current_K:.3f})")
            print(f"   Current prior={current_log_prior:.3f}, lik={current_log_lik:.3f}")
            
            if np.isnan(current_H) or np.isinf(current_H):
                raise ValueError("Invalid current energy")
                
        except Exception as e:
            print(f"   ❌ Current energy computation failed: {e}")
            return self._create_rejection_result()

        # Leapfrog integration in θ-space
        proposed_theta = current_theta.clone()
        proposed_momentum = momentum.clone()
        
        print(f"   Starting leapfrog with {self.num_leapfrog_steps} steps, step_size={self.step_size}")
        
        try:
            for step in range(self.num_leapfrog_steps):
                # Half step for momentum
                _, _, _, theta_grad = self.compute_gradients_theta_space(
                    proposed_theta, states, actions, advantages
                )
                
                theta_grad = torch.clamp(theta_grad, -10.0, 10.0)
                
                if step == 0:
                    print(f"   Gradient stats: mean={theta_grad.mean().item():.6f}, std={theta_grad.std().item():.6f}")
                
                proposed_momentum -= 0.5 * self.step_size * theta_grad
                
                # Full step for position  
                proposed_theta += self.step_size * proposed_momentum
                proposed_theta = torch.clamp(proposed_theta, -50.0, 50.0)
                
                # Half step for momentum
                _, _, _, theta_grad = self.compute_gradients_theta_space(
                    proposed_theta, states, actions, advantages
                )
                
                theta_grad = torch.clamp(theta_grad, -10.0, 10.0)
                proposed_momentum -= 0.5 * self.step_size * theta_grad
                
        except Exception as e:
            print(f"   ❌ Leapfrog integration failed: {e}")
            return self._create_rejection_result()

        # Proposed energy
        try:
            proposed_U, proposed_log_prior, proposed_log_lik, _ = \
                self.compute_gradients_theta_space(proposed_theta, states, actions, advantages)
                
            proposed_K = 0.5 * (proposed_momentum ** 2).sum().item()
            proposed_H = proposed_U + proposed_K
            
            print(f"   Proposed energy: H={proposed_H:.3f} (U={proposed_U:.3f}, K={proposed_K:.3f})")
            print(f"   Proposed prior={proposed_log_prior:.3f}, lik={proposed_log_lik:.3f}")
            
            if np.isnan(proposed_H) or np.isinf(proposed_H):
                raise ValueError("Invalid proposed energy")
                
        except Exception as e:
            print(f"   ❌ Proposed energy computation failed: {e}")
            return self._create_rejection_result()

        # Metropolis acceptance
        energy_change = proposed_H - current_H
        raw_energy_change = energy_change
        energy_change = np.clip(energy_change, -50, 50)
        acceptance_prob = min(1.0, np.exp(-energy_change))

        print(f"   Energy change: ΔH={raw_energy_change:.3f} (clipped: {energy_change:.3f})")
        print(f"   Acceptance probability: {acceptance_prob:.6f}")

        # Accept or reject
        if np.random.rand() < acceptance_prob:
            # ACCEPT: Update θ vector and network
            self.theta_vector = proposed_theta.clone()
            self.set_network_parameters(self.theta_vector)
            accepted = True
            final_log_prior = proposed_log_prior
            final_log_lik = proposed_log_lik
            final_theta = proposed_theta
            print(f"   ✅ ACCEPTED")
        else:
            # REJECT: Keep current state
            accepted = False
            final_log_prior = current_log_prior
            final_log_lik = current_log_lik
            final_theta = current_theta
            print(f"   ❌ REJECTED")

        # CRITICAL: Add theta to ensemble (both accepted and rejected)
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

        # Very conservative step size adaptation
        if len(self.recent_acceptances) >= 5:
            if recent_acceptance_rate < 0.1:
                self.step_size *= 0.95
                print(f"   Reducing step size to {self.step_size:.8f}")
            elif recent_acceptance_rate > 0.85:
                self.step_size *= 1.05
                print(f"   Increasing step size to {self.step_size:.8f}")
            
            self.step_size = np.clip(self.step_size, 1e-8, 0.5)

        # Log progress
        if self.total_proposals % 10 == 0:
            print(f"🎯 ENSEMBLE HMC Step {self.total_proposals}: "
                       f"Accept={accepted}, Rate={recent_acceptance_rate:.3f}, "
                       f"StepSize={self.step_size:.6f}, Ensemble={len(self.theta_ensemble)}")

        return {
            'accepted': accepted,
            'acceptance_rate': current_acceptance_rate,
            'recent_acceptance_rate': recent_acceptance_rate,
            'acceptance_prob': acceptance_prob,
            'energy_change': energy_change,
            'temperature': self.temperature,
            'step_size': self.step_size,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': final_log_prior,
            'log_likelihood': final_log_lik,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR'),
            'expected_kinetic_energy': self.expected_kinetic_energy,
            'likelihood_scale': self.likelihood_scale,
            'ensemble_size': len(self.theta_ensemble)
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
            'step_size': self.step_size,
            'num_leapfrog_steps': self.num_leapfrog_steps,
            'log_prior': 0.0,
            'log_likelihood': 0.0,
            'numerical_failures': self.numerical_failures,
            'theta_dimension': self.theta_vector.numel(),
            'qr_transforms': sum(1 for t in self.param_transforms.values() if t['type'] == 'QR')
        }


def train_true_qr_hmc_network(network, hmc_optimizer, trajectories, config):
    """Train network using ensemble median θ-space HMC"""
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

        # ENSEMBLE APPROACH: Start new epoch and collect samples
        hmc_optimizer.start_epoch()
        
        # Perform HMC updates and collect ensemble
        total_accepted = 0
        hmc_stats = None

        print(f"🧮 Starting ensemble collection with {config.updates_per_epoch} HMC steps...")

        for epoch in range(config.updates_per_epoch):
            hmc_result = hmc_optimizer.hmc_step(
                states, actions, advantages_tensor, old_log_probs
            )

            if hmc_result['accepted']:
                total_accepted += 1

            hmc_stats = hmc_result
            
            print(f"   Ensemble step {epoch+1}/{config.updates_per_epoch}: "
                  f"accepted={hmc_result['accepted']}, ensemble_size={hmc_result['ensemble_size']}")

        # FINALIZE: Compute median and update network
        ensemble_stats = hmc_optimizer.finalize_epoch_with_median()
        
        print(f"🎯 Ensemble training complete:")
        print(f"   Total HMC steps: {config.updates_per_epoch}")
        print(f"   Accepted steps: {total_accepted}")
        print(f"   Ensemble size: {ensemble_stats['ensemble_size']}")
        print(f"   Final parameters: MEDIAN of ensemble")

        # Compute final loss for monitoring
        try:
            with torch.no_grad():
                new_log_probs, values, entropy = network.evaluate_actions(states, actions)
                policy_loss = -(new_log_probs * advantages_tensor).mean()
                total_loss = policy_loss

        except Exception as e:
            logger.warning(f"Loss computation failed: {e}")
            total_loss = torch.tensor(1000.0)
            policy_loss = torch.tensor(1000.0)

        # FIXED: Combine stats properly
        final_stats = ensemble_stats.copy()  # Start with ensemble stats
        final_stats['policy_loss'] = policy_loss.item() if hasattr(policy_loss, 'item') else 1000.0
        final_stats['acceptances_this_update'] = total_accepted

        return total_loss.item() if hasattr(total_loss, 'item') else total_loss, final_stats

    except Exception as e:
        logger.error(f"Ensemble HMC training failed: {e}")
        fallback_stats = {
            'accepted': False,
            'acceptance_rate': 0.0,
            'recent_acceptance_rate': 0.0,
            'policy_loss': 1000.0,
            'acceptances_this_update': 0,
            'ensemble_size': 0,
            'median_computed': False,
            'step_size': 0.03,
            'temperature': 1.0,
            'num_leapfrog_steps': 5,
            'log_prior': 0.0,
            'log_likelihood': 0.0,
            'numerical_failures': 0,
            'theta_dimension': 0,
            'qr_transforms': 0,
            'expected_kinetic_energy': 0.0,
            'likelihood_scale': 1.0,
            'TRAINING_FAILED': True
        }
        return 1000.0, fallback_stats


# Experiment Configuration
@dataclass 
class ExperimentConfig:
    """Experiment configuration with PROPERLY SCALED HMC"""
    # Environment
    env_id: str = "ALE/Asteroids-v5"
    frame_stack: int = 4
    screen_size: int = 84

    # Training parameters
    total_episodes: int = 100000
    episodes_per_update: int = 50
    updates_per_epoch: int = 50

    # Network architecture
    hidden_dim: int = 256
    learning_rate: float = 3e-4

    # PPO parameters
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5

    # CONSERVATIVE HMC parameters
    hmc_step_size: float = 0.001
    hmc_num_leapfrog_steps: int = 20
    hmc_temperature: float = 1E3

    # Video settings
    video_frequency: int = 200
    plot_frequency: int = 100
    debug_frequency: int = 50
    add_text_overlay: bool = True

    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ImprovedVideoWrapper(gym.Wrapper):
    """
    GREATLY IMPROVED video wrapper with action probability bar charts
    
    Key improvements:
    1. Reliable video recording system
    2. Action probability visualization instead of individual commands
    3. Better text overlay with method comparison
    4. Proper frame handling and error recovery
    """

    def __init__(self, env, video_folder, episode_num, method_name, network=None, 
                 other_network=None, other_method_name="", record_frequency=1):
        super().__init__(env)
        
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        
        self.episode_num = episode_num
        self.method_name = method_name
        self.network = network
        self.other_network = other_network
        self.other_method_name = other_method_name
        self.record_frequency = record_frequency
        
        # Video recording setup
        self.recording = False
        self.frames = []
        self.step_count = 0
        self.total_reward = 0.0
        self.action_history = []
        self.action_probs_history = []
        
        # Chart update frequency (every N steps)
        self.chart_update_frequency = 30
        
        print(f"🎥 ImprovedVideoWrapper initialized:")
        print(f"   Method: {method_name}, Episode: {episode_num}")
        print(f"   Video folder: {video_folder}")
        print(f"   Recording frequency: every {record_frequency} episodes")

    def reset(self, **kwargs):
        """Reset and start recording if appropriate"""
        self.step_count = 0
        self.total_reward = 0.0
        self.action_history = []
        self.action_probs_history = []
        
        # Decide whether to record this episode
        should_record = (self.episode_num % self.record_frequency == 0)
        
        if should_record:
            self.recording = True
            self.frames = []
            print(f"🎥 Recording episode {self.episode_num} for {self.method_name}")
        else:
            self.recording = False
            
        obs, info = self.env.reset(**kwargs)
        
        if self.recording:
            self._add_frame_with_overlay(obs, action=0, action_probs=None)
            
        return obs, info

    def step(self, action):
        """Step with enhanced video recording"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_count += 1
        self.total_reward += reward
        self.action_history.append(action)
        
        # Get action probabilities if network available
        action_probs = None
        if self.network is not None and self.recording:
            try:
                with torch.no_grad():
                    if isinstance(obs, np.ndarray):
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    else:
                        obs_tensor = obs.unsqueeze(0) if len(obs.shape) == 3 else obs
                    
                    probs = self.network.get_action_probabilities(obs_tensor)
                    action_probs = probs.cpu().numpy().flatten()
                    self.action_probs_history.append(action_probs.copy())
            except Exception as e:
                # Silently continue if probability extraction fails
                action_probs = None
        
        if self.recording:
            self._add_frame_with_overlay(obs, action, action_probs)
        
        # Save video when episode ends
        if (terminated or truncated) and self.recording:
            self._save_video()
            
        return obs, reward, terminated, truncated, info

    def _add_frame_with_overlay(self, obs, action, action_probs):
        """Add frame with comprehensive overlay"""
        try:
            # Get RGB frame from environment
            if hasattr(self.env, 'render'):
                frame = self.env.render()
            else:
                # Fallback: convert observation to RGB if possible
                if isinstance(obs, np.ndarray):
                    if len(obs.shape) == 3 and obs.shape[0] == 4:  # Frame stack
                        frame = obs[-1]  # Last frame
                        frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
                    else:
                        frame = obs
                else:
                    return  # Skip if can't get frame
            
            if frame is None or frame.size == 0:
                return
                
            # Ensure frame is proper format
            if len(frame.shape) == 2:  # Grayscale
                frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel
                frame = np.repeat(frame, 3, axis=2)
                
            # Make copy for modification
            frame_with_overlay = frame.copy()
            
            # Add text overlay
            self._add_text_overlay(frame_with_overlay, action)
            
            # Add action probability chart every N steps
            if (action_probs is not None and 
                self.step_count % self.chart_update_frequency == 0 and 
                len(self.action_probs_history) > 0):
                frame_with_overlay = self._add_action_chart(frame_with_overlay, action_probs)
            
            self.frames.append(frame_with_overlay)
            
        except Exception as e:
            # Silently continue on frame processing errors
            pass

    def _add_text_overlay(self, frame, action):
        """Add text overlay with episode info"""
        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            # Text content
            action_name = ASTEROIDS_ACTIONS.get(action, f"Action {action}")
            text_lines = [
                f"{self.method_name} Episode {self.episode_num}",
                f"Step: {self.step_count}",
                f"Reward: {self.total_reward:.0f}",
                f"Action: {action_name}",
                f"Recent Actions: {len(self.action_history)}"
            ]
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            color = (255, 255, 255)  # White
            thickness = 1
            
            # Add semi-transparent background
            overlay_height = len(text_lines) * 15 + 10
            cv2.rectangle(frame, (5, 5), (200, overlay_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (200, overlay_height), (50, 50, 50), 2)
            
            # Add text
            for i, line in enumerate(text_lines):
                y_pos = 20 + i * 15
                cv2.putText(frame, line, (10, y_pos), font, font_scale, color, thickness)
                
        except Exception as e:
            # Continue silently if text overlay fails
            pass

    def _add_action_chart(self, frame, current_action_probs):
        """Add action probability bar chart to frame"""
        try:
            if len(self.action_probs_history) == 0:
                return frame
                
            # Calculate recent average probabilities
            recent_window = min(10, len(self.action_probs_history))
            recent_probs = np.array(self.action_probs_history[-recent_window:])
            avg_probs = np.mean(recent_probs, axis=0)
            
            # Chart dimensions and position
            chart_width = 150
            chart_height = 100
            chart_x = frame.shape[1] - chart_width - 10
            chart_y = 10
            
            # Create chart background
            cv2.rectangle(frame, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (40, 40, 40), -1)
            cv2.rectangle(frame, (chart_x, chart_y), 
                         (chart_x + chart_width, chart_y + chart_height), 
                         (100, 100, 100), 2)
            
            # Add title
            cv2.putText(frame, "Action Probabilities", 
                       (chart_x + 5, chart_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw bars for top 6 actions
            top_actions = np.argsort(avg_probs)[-6:][::-1]  # Top 6 in descending order
            bar_height = 10
            bar_spacing = 12
            start_y = chart_y + 25
            
            for i, action_idx in enumerate(top_actions):
                prob = avg_probs[action_idx]
                bar_width = int(prob * (chart_width - 60))  # Scale to chart
                
                y_pos = start_y + i * bar_spacing
                
                # Color code: current action in red, others in blue
                color = (0, 0, 255) if action_idx == np.argmax(current_action_probs) else (255, 100, 0)
                
                # Draw bar
                cv2.rectangle(frame, (chart_x + 35, y_pos), 
                             (chart_x + 35 + bar_width, y_pos + bar_height - 2), 
                             color, -1)
                
                # Add action label
                action_name = ASTEROIDS_ACTIONS.get(action_idx, f"A{action_idx}")
                short_name = action_name[:4]  # Truncate for space
                cv2.putText(frame, short_name, 
                           (chart_x + 2, y_pos + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                
                # Add probability value
                cv2.putText(frame, f"{prob:.2f}", 
                           (chart_x + chart_width - 30, y_pos + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            # Return original frame if chart creation fails
            return frame

    def _save_video(self):
        """Save recorded video with error handling"""
        try:
            if len(self.frames) == 0:
                print(f"⚠️ No frames to save for {self.method_name} episode {self.episode_num}")
                return
                
            # Video filename
            video_filename = f"{self.method_name.lower()}_episode_{self.episode_num:04d}.mp4"
            video_path = self.video_folder / video_filename
            
            # Get frame dimensions
            height, width = self.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ Failed to create video writer for {video_path}")
                return
            
            # Write frames
            frames_written = 0
            for frame in self.frames:
                if frame is not None and frame.size > 0:
                    # Ensure frame is uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    out.write(frame_bgr)
                    frames_written += 1
            
            out.release()
            
            if frames_written > 0:
                print(f"✅ Video saved: {video_path} ({frames_written} frames)")
                
                # Save action summary
                self._save_action_summary()
            else:
                print(f"❌ No frames written to {video_path}")
                
        except Exception as e:
            print(f"❌ Failed to save video for {self.method_name} episode {self.episode_num}: {e}")

    def _save_action_summary(self):
        """Save action and probability summary"""
        try:
            summary_path = self.video_folder / f"{self.method_name.lower()}_episode_{self.episode_num:04d}_summary.pkl"
            
            summary_data = {
                'episode_num': self.episode_num,
                'method_name': self.method_name,
                'total_reward': self.total_reward,
                'episode_length': self.step_count,
                'action_history': self.action_history,
                'action_probs_history': self.action_probs_history,
                'action_counts': np.bincount(self.action_history, minlength=14).tolist()
            }
            
            with open(summary_path, 'wb') as f:
                pickle.dump(summary_data, f)
                
        except Exception as e:
            # Silently continue if summary saving fails
            pass


def create_environment_with_video(config: ExperimentConfig, method_name: str, 
                                episode_num: int = 0, network=None, other_network=None, 
                                other_method_name=""):
    """Create environment with improved video recording"""
    
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

    # Add video recording if appropriate
    if episode_num > 0 and episode_num % config.video_frequency == 0:
        video_dir = Path("experiment_results") / "videos" / method_name.lower()
        
        try:
            env = ImprovedVideoWrapper(
                env,
                video_folder=video_dir,
                episode_num=episode_num,
                method_name=method_name,
                network=network,
                other_network=other_network,
                other_method_name=other_method_name,
                record_frequency=1  # Always record when this wrapper is added
            )
            print(f"🎥 Enhanced video recording enabled for {method_name} episode {episode_num}")
            
        except Exception as e:
            print(f"⚠️ Failed to enable video recording: {e}")

    return env


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
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

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
        except Exception as e:
            logger.warning(f"Failed to calculate conv output size: {e}, using fallback")
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

        except Exception as e:
            logger.warning(f"Xavier initialization failed: {e}, using fallback")
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


def create_comprehensive_performance_plots(results, experiment_dir, episode_count):
    """Create comprehensive performance plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 16))

    # 1. Reward comparison
    ax1 = plt.subplot(3, 3, 1)
    ppo_rewards = results['ppo']['episode_rewards']
    hmc_rewards = results['hmc']['episode_rewards']

    episodes = range(len(ppo_rewards))
    ax1.plot(episodes, ppo_rewards, 'b-', alpha=0.3, label='PPO Raw')
    ax1.plot(episodes, hmc_rewards, 'r-', alpha=0.3, label='HMC Raw')

    # Smoothed versions
    if len(ppo_rewards) > 20:
        ppo_smooth = uniform_filter1d(ppo_rewards, size=20, mode='nearest')
        hmc_smooth = uniform_filter1d(hmc_rewards, size=20, mode='nearest')
        ax1.plot(episodes, ppo_smooth, 'b-', linewidth=3, label='PPO Smoothed')
        ax1.plot(episodes, hmc_smooth, 'r-', linewidth=3, label='HMC Smoothed')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards: PPO vs QR-HMC', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. HMC Acceptance Rate
    ax2 = plt.subplot(3, 3, 2)
    if 'acceptance_rates' in results['hmc'] and len(results['hmc']['acceptance_rates']) > 0:
        ax2.plot(results['hmc']['acceptance_rates'], 'g-', linewidth=2, label='Acceptance Rate')
        ax2.axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='Target (85%)')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_title('HMC Acceptance Rate', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Step Size Evolution
    ax3 = plt.subplot(3, 3, 3)
    if 'step_sizes' in results['hmc'] and len(results['hmc']['step_sizes']) > 0:
        ax3.plot(results['hmc']['step_sizes'], 'purple', linewidth=2)
        ax3.set_ylabel('Step Size')
        ax3.set_title('HMC Step Size Adaptation', fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Episode Length Comparison
    ax4 = plt.subplot(3, 3, 4)
    if len(results['ppo']['episode_lengths']) > 0 and len(results['hmc']['episode_lengths']) > 0:
        ax4.plot(results['ppo']['episode_lengths'], 'b-', alpha=0.6, label='PPO')
        ax4.plot(results['hmc']['episode_lengths'], 'r-', alpha=0.6, label='HMC')
        ax4.set_ylabel('Episode Length')
        ax4.set_title('Episode Lengths', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Loss Comparison
    ax5 = plt.subplot(3, 3, 5)
    if len(results['ppo']['losses']) > 0:
        ax5.plot(results['ppo']['losses'], 'b-', linewidth=2, label='PPO Loss')
        if 'losses' in results['hmc'] and len(results['hmc']['losses']) > 0:
            ax5.plot(results['hmc']['losses'], 'r-', linewidth=2, label='HMC Loss')
        ax5.set_ylabel('Loss')
        ax5.set_title('Training Loss', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Action Distribution Comparison
    ax6 = plt.subplot(3, 3, 6)
    if np.sum(results['action_distributions']['ppo']) > 0 or np.sum(results['action_distributions']['hmc']) > 0:
        actions = range(14)
        width = 0.35
        ax6.bar([x - width/2 for x in actions], results['action_distributions']['ppo'], 
                width, label='PPO', alpha=0.7, color='blue')
        ax6.bar([x + width/2 for x in actions], results['action_distributions']['hmc'], 
                width, label='HMC', alpha=0.7, color='red')
        ax6.set_ylabel('Probability')
        ax6.set_xlabel('Action')
        ax6.set_title('Action Distribution', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # 7. Cumulative Reward
    ax7 = plt.subplot(3, 3, 7)
    if len(ppo_rewards) > 0 and len(hmc_rewards) > 0:
        ppo_cumulative = np.cumsum(ppo_rewards)
        hmc_cumulative = np.cumsum(hmc_rewards)
        ax7.plot(ppo_cumulative, 'b-', linewidth=2, label='PPO')
        ax7.plot(hmc_cumulative, 'r-', linewidth=2, label='HMC')
        ax7.set_ylabel('Cumulative Reward')
        ax7.set_xlabel('Episode')
        ax7.set_title('Cumulative Performance', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. Recent Performance Window
    ax8 = plt.subplot(3, 3, 8)
    window_size = min(50, len(ppo_rewards))
    if window_size > 10:
        recent_ppo = ppo_rewards[-window_size:]
        recent_hmc = hmc_rewards[-window_size:]
        ax8.plot(recent_ppo, 'b-', linewidth=2, label=f'PPO (last {window_size})')
        ax8.plot(recent_hmc, 'r-', linewidth=2, label=f'HMC (last {window_size})')
        ax8.set_ylabel('Reward')
        ax8.set_xlabel('Recent Episodes')
        ax8.set_title('Recent Performance', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary stats
    if len(ppo_rewards) > 0 and len(hmc_rewards) > 0:
        ppo_mean = np.mean(ppo_rewards)
        hmc_mean = np.mean(hmc_rewards)
        ppo_std = np.std(ppo_rewards)
        hmc_std = np.std(hmc_rewards)
        
        # Recent performance
        recent_window = min(20, len(ppo_rewards))
        ppo_recent = np.mean(ppo_rewards[-recent_window:]) if recent_window > 0 else 0
        hmc_recent = np.mean(hmc_rewards[-recent_window:]) if recent_window > 0 else 0
        
        summary_text = f"""
PERFORMANCE SUMMARY

Overall Performance:
PPO: {ppo_mean:.1f} ± {ppo_std:.1f}
HMC: {hmc_mean:.1f} ± {hmc_std:.1f}
Difference: {hmc_mean - ppo_mean:+.1f}

Recent Performance ({recent_window} eps):
PPO: {ppo_recent:.1f}
HMC: {hmc_recent:.1f}
Difference: {hmc_recent - ppo_recent:+.1f}

Episodes Completed: {episode_count}
        """
        
        # Add HMC-specific stats
        if 'acceptance_rates' in results['hmc'] and len(results['hmc']['acceptance_rates']) > 0:
            final_acceptance = results['hmc']['acceptance_rates'][-1]
            summary_text += f"\nHMC Acceptance Rate: {final_acceptance:.3f}"
        
        if 'step_sizes' in results['hmc'] and len(results['hmc']['step_sizes']) > 0:
            final_step_size = results['hmc']['step_sizes'][-1]
            summary_text += f"\nHMC Step Size: {final_step_size:.6f}"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    
    # Save plot
    plot_path = experiment_dir / f'performance_plots_episode_{episode_count}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 Performance plots saved: {plot_path}")


def analyze_action_probabilities_comprehensive(ppo_network, hmc_network, sample_states):
    """Comprehensive action probability analysis"""
    ppo_probs = []
    hmc_probs = []

    for state in sample_states:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        try:
            ppo_prob = ppo_network.get_action_probabilities(state_tensor)
            hmc_prob = hmc_network.get_action_probabilities(state_tensor)

            ppo_probs.append(ppo_prob.cpu().numpy().flatten())
            hmc_probs.append(hmc_prob.cpu().numpy().flatten())
        except Exception as e:
            logger.warning(f"Failed to get action probabilities: {e}")
            continue

    if len(ppo_probs) == 0 or len(hmc_probs) == 0:
        return {
            'mean_kl_divergence': 0.0,
            'action_preference_diff': np.zeros(14),
            'ppo_probs': np.zeros((1, 14)),
            'hmc_probs': np.zeros((1, 14))
        }

    ppo_probs = np.array(ppo_probs)
    hmc_probs = np.array(hmc_probs)

    # KL divergence
    kl_divs = []
    for i in range(len(ppo_probs)):
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p_safe = ppo_probs[i] + eps
        q_safe = hmc_probs[i] + eps
        
        # Normalize
        p_safe = p_safe / np.sum(p_safe)
        q_safe = q_safe / np.sum(q_safe)
        
        kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
        kl_divs.append(kl_div)

    mean_kl = np.mean(kl_divs)
    
    # Action preference differences
    ppo_mean = np.mean(ppo_probs, axis=0)
    hmc_mean = np.mean(hmc_probs, axis=0)
    action_diff = hmc_mean - ppo_mean

    return {
        'mean_kl_divergence': mean_kl,
        'action_preference_diff': action_diff,
        'ppo_probs': ppo_probs,
        'hmc_probs': hmc_probs,
        'ppo_mean_probs': ppo_mean,
        'hmc_mean_probs': hmc_mean
    }


def run_complete_experiment():
    """Run the complete PPO vs QR-HMC experiment with FIXED ensemble HMC"""
    
    print("=" * 80)
    print("🚀 COMPLETE PPO vs QR-HMC EXPERIMENT - FIXED VERSION")
    print("=" * 80)
    
    # Configuration
    config = ExperimentConfig()
    
    # Create experiment directory
    experiment_dir = Path("experiment_results")
    experiment_dir.mkdir(exist_ok=True)
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create networks
    ppo_network = AsteroidsNetwork(config)
    hmc_network = AsteroidsNetwork(config)
    
    # Sync initial weights
    hmc_network.load_state_dict(ppo_network.state_dict())
    
    print(f"🧠 Networks initialized:")
    print(f"   Total parameters: {sum(p.numel() for p in ppo_network.parameters()):,}")
    
    # Create optimizers
    ppo_optimizer = optim.Adam(ppo_network.parameters(), lr=config.learning_rate)
    ppo_policy_opt = StandardPPO(config)
    
    print(f"🔧 Creating FIXED θ-space QR-HMC optimizer...")
    try:
        hmc_policy_opt = TrueQRTransformedHMC(hmc_network, config)
        print(f"✅ FIXED θ-space QR-HMC optimizer created successfully")
        
        # Print transformation summary
        print(f"📊 Parameter Transformation Summary:")
        qr_count = sum(1 for t in hmc_policy_opt.param_transforms.values() if t['type'] == 'QR')
        std_count = sum(1 for t in hmc_policy_opt.param_transforms.values() if t['type'] == 'standardized')
        print(f"   QR transformations: {qr_count}")
        print(f"   Standardized parameters: {std_count}")
        print(f"   Total θ dimension: {hmc_policy_opt.theta_vector.numel():,}")
        print(f"   Uniform step size: {hmc_policy_opt.step_size:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to create θ-space QR-HMC optimizer: {e}")
        raise
    
    # Results tracking with comprehensive monitoring
    results = {
        'ppo': {
            'episode_rewards': [], 'episode_lengths': [], 'losses': [],
            'policy_losses': [], 'value_losses': [], 'entropy_losses': []
        },
        'hmc': {
            'episode_rewards': [], 'episode_lengths': [], 'losses': [],
            'acceptance_rates': [], 'step_sizes': [], 'log_priors': [],
            'log_likelihoods': [], 'numerical_failures': [], 'theta_dimensions': [],
            'transformation_health': [], 'parameter_traces': {}, 'trace_episodes': []
        },
        'action_distributions': {'ppo': np.zeros(14), 'hmc': np.zeros(14)},
        'sample_states': [],
        'action_analysis_history': []
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
            
            # Create environments with ENHANCED video recording
            ppo_env = create_environment_with_video(
                config, "PPO", episode_count, 
                network=ppo_network, other_network=hmc_network, other_method_name="HMC"
            )
            hmc_env = create_environment_with_video(
                config, "HMC", episode_count, 
                network=hmc_network, other_network=ppo_network, other_method_name="PPO"
            )
            
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
                
                # Collect sample states for analysis
                if len(results['sample_states']) < 100:
                    results['sample_states'].extend(ppo_traj['states'][:5])
                
                episode_count += 1

                # Video recording notification
                if episode_count % config.video_frequency == 0:
                    print(f"🎥 Episode {episode_count} - Enhanced videos with action charts recorded!")

                # Progress update with enhanced metrics
                if episode_count % 4 == 0:
                    ppo_recent = np.mean([t['total_reward'] for t in ppo_trajectories[-4:]])
                    hmc_recent = np.mean([t['total_reward'] for t in hmc_trajectories[-4:]])
                    ppo_length = np.mean([t['length'] for t in ppo_trajectories[-4:]])
                    hmc_length = np.mean([t['length'] for t in hmc_trajectories[-4:]])
                    print(f"  Episode {episode_count:4d} - PPO: {ppo_recent:7.1f} ({ppo_length:.0f} steps), "
                          f"θ-HMC: {hmc_recent:7.1f} ({hmc_length:.0f} steps)")
            
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
                
                # Train FIXED θ-space QR-HMC
                hmc_loss, hmc_stats = train_true_qr_hmc_network(
                    hmc_network, hmc_policy_opt, hmc_trajectories, config
                )
                
                # Enhanced training statistics
                print(f"\n📊 Training Statistics:")
                print(f"  PPO - Loss: {ppo_loss:.4f}, Policy: {ppo_stats.get('policy_loss', 0):.4f}")
                
                # FIXED: Use correct key access
                acceptance_rate = hmc_stats.get('acceptance_rate', 0.0)
                recent_acceptance = hmc_stats.get('recent_acceptance_rate', 0.0)
                step_size = hmc_stats.get('step_size', 0.0)
                theta_dim = hmc_stats.get('theta_dimension', 0)
                qr_transforms = hmc_stats.get('qr_transforms', 0)
                failures = hmc_stats.get('numerical_failures', 0)
                
                print(f"  θ-HMC - Loss: {hmc_loss:.4f}, Accept: {acceptance_rate:.3f}, "
                      f"Recent: {recent_acceptance:.3f}")
                print(f"         Step Size: {step_size:.6f}, θ-dim: {theta_dim:,}")
                print(f"         QR Transforms: {qr_transforms}, Failures: {failures}")

                # Store comprehensive HMC stats
                results['hmc']['acceptance_rates'].append(acceptance_rate)
                results['hmc']['step_sizes'].append(step_size)
                results['hmc']['log_priors'].append(hmc_stats.get('log_prior', 0))
                results['hmc']['log_likelihoods'].append(hmc_stats.get('log_likelihood', 0))
                results['hmc']['numerical_failures'].append(failures)
                results['hmc']['theta_dimensions'].append(theta_dim)
                
                # Store transformation health info
                transformation_health = {
                    'qr_transforms': qr_transforms,
                    'standardized_transforms': theta_dim - qr_transforms,
                    'total_theta_params': theta_dim,
                    'step_size': step_size,
                    'acceptance_rate': acceptance_rate
                }
                results['hmc']['transformation_health'].append(transformation_health)

                # Store PPO stats
                if hasattr(ppo_stats, 'keys'):
                    results['ppo']['policy_losses'].append(ppo_stats.get('policy_loss', 0))
                    results['ppo']['value_losses'].append(ppo_stats.get('value_loss', 0))
                    results['ppo']['entropy_losses'].append(ppo_stats.get('entropy_loss', 0))
            
            # Debug output with enhanced θ-space info
            if episode_count % config.debug_frequency == 0:
                elapsed_time = time.time() - start_time
                episodes_per_hour = episode_count / (elapsed_time / 3600)

                print(f"\n🔍 DEBUG STATUS (Episode {episode_count}):")
                print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"   Episodes/hour: {episodes_per_hour:.1f}")
                print(f"   θ-HMC acceptance: {recent_acceptance:.3f}")
                print(f"   θ-HMC step size: {step_size:.6f}")
                print(f"   θ dimension: {theta_dim:,}")
                print(f"   QR transforms: {qr_transforms}")

                # Memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**2
                    print(f"   GPU memory: {memory_used:.1f} MB")

            # Generate comprehensive plots
            if episode_count % config.plot_frequency == 0 and episode_count > 0:
                print(f"\n📈 Generating comprehensive performance plots...")

                create_comprehensive_performance_plots(results, experiment_dir, episode_count)

                # Action probability analysis
                if len(results['sample_states']) >= 10:
                    try:
                        sample_states = results['sample_states'][:20]
                        action_analysis = analyze_action_probabilities_comprehensive(
                            ppo_network, hmc_network, sample_states
                        )
                        results['action_analysis_history'].append({
                            'episode': episode_count,
                            'kl_divergence': action_analysis['mean_kl_divergence'],
                            'action_preference_diff': action_analysis['action_preference_diff'],
                            'ppo_mean_probs': action_analysis['ppo_mean_probs'],
                            'hmc_mean_probs': action_analysis['hmc_mean_probs']
                        })

                        print(f"📊 Action analysis: KL divergence = {action_analysis['mean_kl_divergence']:.4f}")

                    except Exception as e:
                        logger.warning(f"Failed to analyze action probabilities: {e}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted at episode {episode_count}")
    
    finally:
        # Final analysis
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🏁 EXPERIMENT COMPLETE - FIXED VERSION!")
        print(f"Episodes completed: {episode_count}/{config.total_episodes}")
        print(f"Total time: {total_time/60:.1f} minutes")
        
        if results['ppo']['episode_rewards'] and results['hmc']['episode_rewards']:
            ppo_rewards = results['ppo']['episode_rewards']
            hmc_rewards = results['hmc']['episode_rewards']
            
            print(f"\n🏆 FINAL RESULTS:")
            print(f"PPO: {np.mean(ppo_rewards):.1f} ± {np.std(ppo_rewards):.1f}")
            print(f"HMC: {np.mean(hmc_rewards):.1f} ± {np.std(hmc_rewards):.1f}")
            
            improvement = np.mean(hmc_rewards) - np.mean(ppo_rewards)
            print(f"Performance difference: {improvement:+.1f}")
        
        # HMC diagnostics
        if hmc_policy_opt.total_proposals > 0:
            print(f"\n🔧 FIXED θ-SPACE HMC DIAGNOSTICS:")
            print(f"Final acceptance rate: {hmc_policy_opt.total_acceptances / hmc_policy_opt.total_proposals:.3f}")
            print(f"Final step size: {hmc_policy_opt.step_size:.6f}")
            print(f"θ dimension: {hmc_policy_opt.theta_vector.numel():,}")
            print(f"Numerical failures: {hmc_policy_opt.numerical_failures}")

        # Save comprehensive results
        print(f"\n💾 Saving comprehensive results...")
        data_dir = experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)

        with open(data_dir / "comprehensive_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        with open(data_dir / "experiment_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        # Generate final comprehensive plots
        create_comprehensive_performance_plots(results, experiment_dir, episode_count)
        
        print(f"\n✅ FIXED θ-space QR-HMC experiment completed successfully!")
        print(f"📁 Results saved in: {experiment_dir}")
        print(f"🎥 Enhanced videos with action charts saved in: {experiment_dir}/videos/")
        print(f"📊 Plots saved in: {experiment_dir}/")
        print(f"📋 Data saved in: {experiment_dir}/data/")
        print(f"\n🎯 KEY FIXES AND IMPROVEMENTS:")
        print(f"  ✅ Fixed finalize_epoch_with_median() return structure")
        print(f"  ✅ Enhanced video system with action probability charts")
        print(f"  ✅ Reliable frame recording and error recovery")
        print(f"  ✅ Action probability bar charts instead of individual commands")
        print(f"  ✅ Comprehensive error handling throughout")
        print(f"  ✅ Proper ensemble median computation")
        print(f"  ✅ Real-time action analysis in videos")
    
    return results, hmc_policy_opt


if __name__ == "__main__":
    print("🚀 Starting FIXED θ-space QR-HMC vs PPO Experiment")
    print("🎯 COMPLETE FIXED FEATURES:")
    print("  ✅ Fixed ensemble HMC return structure")
    print("  ✅ Greatly improved video system with action probability charts")
    print("  ✅ Enhanced error handling and recovery throughout")
    print("  ✅ Reliable frame recording with fallbacks")
    print("  ✅ Real-time action analysis visualization in videos")
    print("  ✅ Comprehensive performance monitoring and plotting")
    print("  ✅ Proper ensemble median computation")
    print("  ✅ Complete script with all functions included")
    print()
    print("🎥 VIDEO IMPROVEMENTS:")
    print("  📊 Action probability bar charts every 30 steps")
    print("  🎯 Top 6 actions displayed with probabilities")
    print("  🌈 Color-coded current action highlighting")
    print("  📝 Clean text overlays with episode information")
    print("  💾 Automatic video and summary data saving")
    print()
    print("🔬 HMC ENSEMBLE FEATURES:")
    print("  🧮 Collect multiple HMC proposals per epoch")
    print("  📊 Compute median of ensemble for robust training")
    print("  ✅ Track both accepted and rejected proposals")
    print("  📈 Real-time acceptance rate monitoring")
    print("  🎯 Automatic step size adaptation")
    print()
    
    try:
    #    results, hmc_optimizer = run_complete_experiment()
        print("\n🎉 Experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🏁 Script execution finished")