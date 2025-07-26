#!/usr/bin/env python3
"""
Hybrid Asteroids AI: Combining Neuroevolution, LLMs, and Policy Gradients
Designed for Google Colab with GPU support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics
from collections import deque
import random
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pickle
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register ALE environments
gym.register_envs(ale_py)

@dataclass
class HybridConfig:
    """Configuration for hybrid training approach"""
    
    # Environment
    env_id: str = "ALE/Asteroids-v5"
    frame_stack: int = 4
    screen_size: int = 84
    
    # Neuroevolution
    population_size: int = 40
    elite_size: int = 5
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    
    # LLM Configuration
    use_llm_guidance: bool = True
    llm_model_name: str = "microsoft/phi-2"  # Small, efficient model
    llm_quantization: bool = True  # 4-bit quantization for efficiency
    llm_update_frequency: int = 10  # Generations between LLM strategy updates
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    
    # Policy Gradient
    use_policy_gradient: bool = True
    pg_learning_rate: float = 1e-4
    pg_gamma: float = 0.99
    pg_episodes_per_update: int = 5
    pg_entropy_coef: float = 0.01
    pg_value_coef: float = 0.5
    pg_gae_lambda: float = 0.95
    pg_clip_epsilon: float = 0.2  # PPO clipping
    
    # Hybrid Training
    hybrid_mode: str = "sequential"  # "sequential", "parallel", or "mixed"
    neuroevolution_weight: float = 0.5
    pg_weight: float = 0.3
    llm_weight: float = 0.2
    
    # Training
    generations: int = 500
    episodes_per_eval: int = 3
    save_frequency: int = 20
    
    # Video Recording - HIGH QUALITY
    video_frequency: int = 10  # Record every N generations
    video_episodes: int = 3    # Episodes per recording session
    video_resolution: int = 210  # Full Atari resolution
    video_fps: int = 60        # Maximum FPS for smooth playback
    record_frame_skip: int = 1  # NO FRAME SKIPPING in recordings
    
    # GPU/Performance
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 4
    
    # Paths
    save_dir: str = "hybrid_asteroids_ai"
    
    def __post_init__(self):
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("💻 Using CPU")


class LLMStrategyGuide:
    """LLM-based strategy generation and game state analysis"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.strategies = []
        self.current_strategy = None
        self.model = None
        self.tokenizer = None
        
        if config.use_llm_guidance:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM with quantization for efficiency"""
        logger.info(f"🤖 Loading LLM: {self.config.llm_model_name}")
        
        try:
            # Import transformers here to avoid dependency issues
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            if self.config.llm_quantization and self.device.type == "cuda":
                # 4-bit quantization for GPU efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Standard loading for CPU or non-quantized
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm_model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ LLM loaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Transformers not available: {e}")
            logger.info("Install with: pip install transformers accelerate bitsandbytes")
            self.config.use_llm_guidance = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load LLM: {e}")
            logger.info("Continuing without LLM guidance")
            self.config.use_llm_guidance = False
    
    def analyze_performance(self, fitness_history: List[float], 
                          episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze performance and suggest improvements"""
        
        if not self.config.use_llm_guidance or self.model is None:
            return {"strategy": "baseline", "insights": []}
        
        # Prepare context
        recent_fitness = fitness_history[-10:] if len(fitness_history) > 10 else fitness_history
        avg_recent = np.mean(recent_fitness) if recent_fitness else 0
        
        # Create prompt
        prompt = f"""You are an AI coach for the classic Asteroids game. Analyze this performance data and provide strategic advice.

Recent Performance:
- Average Score: {avg_recent:.0f}
- Trend: {'improving' if len(recent_fitness) > 1 and recent_fitness[-1] > recent_fitness[0] else 'plateauing'}
- Deaths: Mostly from {episode_data.get('death_cause', 'collisions')}

Current Issues:
{self._identify_issues(episode_data)}

Provide a concise strategy in 3-4 bullet points focusing on:
1. Movement patterns
2. Shooting strategy  
3. Risk management

Strategy:"""

        try:
            # Generate strategy
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.llm_max_tokens,
                    temperature=self.config.llm_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            strategy_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            strategy_text = strategy_text[len(prompt):].strip()
            
            # Parse strategy into actionable items
            strategy_items = self._parse_strategy(strategy_text)
            
            return {
                "strategy": strategy_text,
                "insights": strategy_items,
                "movement_bias": self._extract_movement_bias(strategy_items),
                "aggression_level": self._extract_aggression_level(strategy_items),
                "safety_priority": self._extract_safety_priority(strategy_items)
            }
            
        except Exception as e:
            logger.warning(f"LLM strategy generation failed: {e}")
            return {"strategy": "baseline", "insights": []}
    
    def _identify_issues(self, episode_data: Dict) -> str:
        """Identify key issues from episode data"""
        issues = []
        
        if episode_data.get('avg_lifetime', 0) < 30:
            issues.append("- Very short survival time")
        if episode_data.get('shots_fired', 0) > episode_data.get('asteroids_destroyed', 0) * 3:
            issues.append("- Poor shooting accuracy")
        if episode_data.get('close_calls', 0) > 5:
            issues.append("- Too many risky maneuvers")
            
        return "\n".join(issues) if issues else "- Performance is stable"
    
    def _parse_strategy(self, strategy_text: str) -> List[str]:
        """Parse LLM output into strategy items"""
        lines = strategy_text.split('\n')
        strategy_items = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the line
                clean_line = line.lstrip('0123456789.-•').strip()
                if clean_line:
                    strategy_items.append(clean_line)
        
        return strategy_items[:5]  # Limit to 5 items
    
    def _extract_movement_bias(self, items: List[str]) -> Dict[str, float]:
        """Extract movement preferences from strategy"""
        bias = {"forward": 0.5, "rotation": 0.5, "thrust": 0.5}
        
        for item in items:
            item_lower = item.lower()
            if "aggressive" in item_lower or "forward" in item_lower:
                bias["forward"] += 0.2
            if "defensive" in item_lower or "dodge" in item_lower:
                bias["rotation"] += 0.2
            if "speed" in item_lower or "fast" in item_lower:
                bias["thrust"] += 0.2
                
        # Normalize
        for key in bias:
            bias[key] = min(1.0, bias[key])
            
        return bias
    
    def _extract_aggression_level(self, items: List[str]) -> float:
        """Extract aggression level from strategy"""
        aggression = 0.5
        
        for item in items:
            item_lower = item.lower()
            if any(word in item_lower for word in ["aggressive", "attack", "destroy"]):
                aggression += 0.15
            if any(word in item_lower for word in ["defensive", "careful", "avoid"]):
                aggression -= 0.15
                
        return max(0.1, min(1.0, aggression))
    
    def _extract_safety_priority(self, items: List[str]) -> float:
        """Extract safety priority from strategy"""
        safety = 0.5
        
        for item in items:
            item_lower = item.lower()
            if any(word in item_lower for word in ["safe", "avoid", "careful", "defensive"]):
                safety += 0.2
            if any(word in item_lower for word in ["risk", "aggressive", "close"]):
                safety -= 0.1
                
        return max(0.1, min(1.0, safety))


class HybridAsteroidsNetwork(nn.Module):
    """Hybrid network combining neuroevolution with policy gradient capabilities"""
    
    def __init__(self, config: HybridConfig, n_actions: int = 14):
        super().__init__()
        self.config = config
        self.n_actions = n_actions
        self.device = torch.device(config.device)
        
        # CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, config.frame_stack, config.screen_size, config.screen_size)
            conv_out = self.conv_layers(dummy)
            self.conv_out_size = conv_out.numel()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Policy head (for both neuroevolution and PG)
        self.policy_head = nn.Sequential(
            nn.Linear(256 + 3, 128),  # +3 for LLM strategy embedding
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Value head (for policy gradient)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # LLM strategy embedding
        self.strategy_embedding = nn.Parameter(torch.zeros(3))
        
        # Evolution tracking
        self.fitness = 0.0
        self.generation = 0
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x, return_value=False):
        """Forward pass with optional value output"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        if x.device != self.device:
            x = x.to(self.device)
            
        # Normalize
        x = x.float() / 255.0
        
        # Handle batch dimensions
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # CNN features
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Shared features
        features = self.feature_extractor(conv_out)
        
        # Add strategy embedding
        batch_size = features.size(0)
        strategy_emb = self.strategy_embedding.unsqueeze(0).expand(batch_size, -1)
        policy_features = torch.cat([features, strategy_emb], dim=1)
        
        # Policy output
        policy_logits = self.policy_head(policy_features)
        
        if return_value:
            value = self.value_head(features)
            return policy_logits, value
        else:
            return policy_logits
    
    def get_action(self, obs, deterministic=False):
        """Get action for neuroevolution"""
        with torch.no_grad():
            logits = self.forward(obs)
            
            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()
    
    def get_action_pg(self, obs):
        """Get action for policy gradient with log prob"""
        logits, value = self.forward(obs, return_value=True)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value
    
    def update_strategy_embedding(self, llm_strategy: Dict[str, Any]):
        """Update strategy embedding based on LLM guidance"""
        if "movement_bias" in llm_strategy:
            movement_bias = llm_strategy["movement_bias"]
            self.strategy_embedding.data[0] = movement_bias.get("forward", 0.5) - 0.5
        if "aggression_level" in llm_strategy:
            self.strategy_embedding.data[1] = llm_strategy["aggression_level"] - 0.5
        if "safety_priority" in llm_strategy:
            self.strategy_embedding.data[2] = llm_strategy["safety_priority"] - 0.5
    
    def mutate(self):
        """Neuroevolution mutation"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * self.config.mutation_strength
                    param.add_(noise)
        
        self.fitness = 0.0
    
    def crossover(self, other):
        """Neuroevolution crossover"""
        offspring = HybridAsteroidsNetwork(self.config, self.n_actions)
        offspring.load_state_dict(self.state_dict())
        
        with torch.no_grad():
            for (child_param, parent2_param) in zip(offspring.parameters(), other.parameters()):
                mask = torch.rand_like(child_param) < 0.5
                child_param[mask] = parent2_param[mask]
        
        offspring.generation = max(self.generation, other.generation) + 1
        return offspring


class PolicyGradientTrainer:
    """PPO-based policy gradient trainer"""
    
    def __init__(self, network: HybridAsteroidsNetwork, config: HybridConfig):
        self.network = network
        self.config = config
        self.optimizer = optim.Adam(network.parameters(), lr=config.pg_learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def collect_experience(self, env, episodes: int):
        """Collect experience for policy gradient update"""
        self.clear_buffers()
        total_reward = 0
        
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(10000):  # Max steps
                # Get action with policy gradient
                action, log_prob, value = self.network.get_action_pg(state)
                
                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.values.append(value.squeeze().item())
                self.log_probs.append(log_prob)
                self.dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward
    
    def compute_gae(self):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        # Convert to tensors
        rewards = torch.FloatTensor(self.rewards).to(self.network.device)
        values = torch.FloatTensor(self.values).to(self.network.device)
        dones = torch.FloatTensor(self.dones).to(self.network.device)
        
        # Compute advantages using GAE
        advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.pg_gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.config.pg_gamma * self.config.pg_gae_lambda * (1 - dones[t]) * advantage
            
            returns.insert(0, advantage + values[t])
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages).to(self.network.device)
        returns = torch.FloatTensor(returns).to(self.network.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """PPO update"""
        if len(self.states) == 0:
            return 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Convert experiences to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.network.device)
        actions = torch.LongTensor(self.actions).to(self.network.device)
        old_log_probs = torch.stack(self.log_probs).to(self.network.device)
        
        # Multiple epochs of PPO
        total_loss = 0
        for _ in range(4):  # PPO epochs
            # Get current policy
            logits, values = self.network.forward(states, return_value=True)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # New log probs
            new_log_probs = dist.log_prob(actions)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.pg_clip_epsilon, 1 + self.config.pg_clip_epsilon) * advantages
            
            # Losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.pg_value_coef * value_loss - 
                   self.config.pg_entropy_coef * entropy_loss)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / 4
    
    def clear_buffers(self):
        """Clear experience buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class FrameStack(gym.Wrapper):
    """Frame stacking wrapper"""
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(num_stack, obs_shape[0], obs_shape[1]),
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


class HybridTrainer:
    """Main trainer combining all three approaches"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.llm_guide = LLMStrategyGuide(config)
        self.population = []
        self.best_network = None
        self.generation = 0
        
        # Tracking
        self.fitness_history = []
        self.pg_reward_history = []
        self.hybrid_score_history = []
        
        # Create environment for testing
        self.env = self._create_env()
        
        logger.info(f"🚀 Hybrid Trainer initialized")
        logger.info(f"   Mode: {config.hybrid_mode}")
        logger.info(f"   Weights: NE={config.neuroevolution_weight}, PG={config.pg_weight}, LLM={config.llm_weight}")
    
    def record_high_quality_videos(self, generation):
        """Record HIGH QUALITY videos with NO frame skipping"""
        logger.info(f"🎬 Recording HIGH QUALITY videos for generation {generation}")
        
        video_dir = self.save_dir / f"videos_gen_{generation:04d}"
        video_dir.mkdir(exist_ok=True, parents=True)
        
        # Use the best network
        best_network = self.population[0]
        
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available for video recording. Install with: pip install opencv-python")
            return
        
        for episode_num in range(self.config.video_episodes):
            logger.info(f"  Recording episode {episode_num + 1}/{self.config.video_episodes}")
            
            # Create HIGH QUALITY environment with NO frame skipping
            env = gym.make(
                self.config.env_id,
                render_mode="rgb_array",
                frameskip=1,  # ABSOLUTE MINIMUM frameskip in base environment
                repeat_action_probability=0.0  # No action repeat
            )
            
            # Wrap with minimal preprocessing for recording
            env = AtariPreprocessing(
                env,
                noop_max=0,  # No random no-ops
                frame_skip=1,  # NO additional frame skipping!
                screen_size=self.config.video_resolution,  # Full resolution
                terminal_on_life_loss=False,  # Full episodes
                grayscale_obs=False,  # Keep RGB for video
                grayscale_newaxis=False,
                scale_obs=False  # Keep original pixel values
            )
            
            # Manual video recording with cv2
            video_path = video_dir / f"episode_{episode_num:02d}.mp4"
            
            # Video writer setup - high quality settings
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = None
            
            # Frame buffer for network input (grayscale conversion)
            gray_frames = deque(maxlen=self.config.frame_stack)
            
            obs, info = env.reset()
            episode_reward = 0
            frame_count = 0
            
            # Initialize video writer with first frame
            if video_writer is None:
                h, w = obs.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.config.video_fps,  # 60 FPS for smooth playback
                    (w, h)
                )
            
            # Initialize grayscale frames
            gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            gray_obs = cv2.resize(gray_obs, (self.config.screen_size, self.config.screen_size))
            for _ in range(self.config.frame_stack):
                gray_frames.append(gray_obs)
            
            while True:
                # Write EVERY frame to video (no skipping!)
                video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                
                # Prepare network input (grayscale, resized, stacked)
                gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                gray_obs = cv2.resize(gray_obs, (self.config.screen_size, self.config.screen_size))
                gray_frames.append(gray_obs)
                
                # Get action from network
                network_input = np.array(gray_frames)
                action = best_network.get_action(network_input, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                frame_count += 1
                
                if terminated or truncated:
                    break
            
            # Clean up
            video_writer.release()
            env.close()
            
            logger.info(f"    ✅ Episode {episode_num + 1}: Score={episode_reward:.0f}, Frames={frame_count}")
        
        logger.info(f"🎬 Videos saved to: {video_dir}")
    
    def _create_env(self):
        """Create environment for training (with frame skipping for speed)"""
        # Create environment with frameskip disabled at the base level
        env = gym.make(self.config.env_id, render_mode=None, frameskip=1)
        
        # Apply preprocessing with our own frame skipping
        env = AtariPreprocessing(
            env,
            frame_skip=4,  # Training uses frame skip for speed
            screen_size=self.config.screen_size,
            grayscale_obs=True,
            scale_obs=False
        )
        
        # Frame stacking
        env = FrameStack(env, self.config.frame_stack)
        return env
    
    def initialize_population(self):
        """Initialize population of networks"""
        logger.info("Initializing hybrid population...")
        
        for i in range(self.config.population_size):
            network = HybridAsteroidsNetwork(self.config)
            network.generation = 0
            self.population.append(network)
        
        logger.info(f"✅ Population initialized: {len(self.population)} networks")
    
    def evaluate_neuroevolution(self):
        """Evaluate population using neuroevolution"""
        logger.info(f"🧬 Evaluating generation {self.generation} (Neuroevolution)")
        
        for i, network in enumerate(self.population):
            total_reward = 0
            
            for episode in range(self.config.episodes_per_eval):
                state, _ = self.env.reset()
                episode_reward = 0
                
                for _ in range(10000):  # Max steps
                    action = network.get_action(state)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                total_reward += episode_reward
            
            network.fitness = total_reward / self.config.episodes_per_eval
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(self.population)}")
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = self.population[0].fitness
        
        logger.info(f"  Best fitness: {best_fitness:.1f}")
        return best_fitness
    
    def train_policy_gradient(self, network: HybridAsteroidsNetwork, episodes: int = 5):
        """Train network using policy gradients"""
        pg_trainer = PolicyGradientTrainer(network, self.config)
        
        # Collect experience and update
        total_reward = pg_trainer.collect_experience(self.env, episodes)
        loss = pg_trainer.update()
        
        return total_reward / episodes, loss
    
    def create_next_generation(self):
        """Create next generation using evolution"""
        next_population = []
        
        # Elite selection
        for i in range(self.config.elite_size):
            elite = self.population[i]
            elite_copy = HybridAsteroidsNetwork(self.config)
            elite_copy.load_state_dict(elite.state_dict())
            elite_copy.generation = self.generation + 1
            next_population.append(elite_copy)
        
        # Create offspring
        while len(next_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            offspring = parent1.crossover(parent2)
            
            # Mutation
            offspring.mutate()
            
            next_population.append(offspring)
        
        self.population = next_population
        self.generation += 1
    
    def _tournament_selection(self, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(self.population[:20], min(tournament_size, 20))
        return max(tournament, key=lambda x: x.fitness)
    
    def update_with_llm_strategy(self):
        """Update networks with LLM-generated strategies"""
        if not self.config.use_llm_guidance:
            return
        
        logger.info("🤖 Generating LLM strategies...")
        
        # Analyze recent performance
        episode_data = {
            "avg_lifetime": 45,  # Placeholder - would track actual data
            "death_cause": "collision",
            "shots_fired": 200,
            "asteroids_destroyed": 50,
            "close_calls": 10
        }
        
        llm_strategy = self.llm_guide.analyze_performance(
            self.fitness_history[-20:] if len(self.fitness_history) > 20 else self.fitness_history,
            episode_data
        )
        
        # Update top performers with strategy
        for network in self.population[:10]:
            network.update_strategy_embedding(llm_strategy)
        
        insights = llm_strategy.get('insights', ['No strategy'])
        logger.info(f"  Strategy insights: {insights[:2]}")
    
    def train_hybrid(self):
        """Main hybrid training loop"""
        logger.info("🚀 Starting Hybrid Training")
        logger.info(f"   Components: {'✅ Neuroevolution' if True else '❌ Neuroevolution'}, "
                   f"{'✅ Policy Gradient' if self.config.use_policy_gradient else '❌ Policy Gradient'}, "
                   f"{'✅ LLM Guidance' if self.config.use_llm_guidance else '❌ LLM Guidance'}")
        
        try:
            self.initialize_population()
            
            for generation in range(self.config.generations):
                logger.info(f"\n{'='*60}")
                logger.info(f"Generation {generation + 1}/{self.config.generations}")
                
                # 1. Neuroevolution evaluation
                ne_fitness = self.evaluate_neuroevolution()
                self.fitness_history.append(ne_fitness)
                
                # Track best network after first evaluation
                if self.best_network is None:
                    self.best_network = self.population[0]
                
                # 2. Policy gradient fine-tuning for top performers
                if self.config.use_policy_gradient and generation % 5 == 0:
                    logger.info("📈 Policy gradient fine-tuning...")
                    
                    for i in range(min(5, len(self.population))):
                        network = self.population[i]
                        try:
                            pg_reward, pg_loss = self.train_policy_gradient(network, episodes=3)
                            self.pg_reward_history.append(pg_reward)
                            
                            if i == 0:
                                logger.info(f"  Best network PG reward: {pg_reward:.1f}, Loss: {pg_loss:.4f}")
                        except Exception as e:
                            logger.warning(f"Policy gradient training failed for network {i}: {e}")
                
                # 3. LLM strategy update
                if generation % self.config.llm_update_frequency == 0:
                    try:
                        self.update_with_llm_strategy()
                    except Exception as e:
                        logger.warning(f"LLM strategy update failed: {e}")
                
                # 4. Calculate hybrid scores
                self._calculate_hybrid_scores()
                
                # 5. Record high-quality videos periodically
                if generation % self.config.video_frequency == 0 or generation == self.config.generations - 1:
                    try:
                        self.record_high_quality_videos(generation)
                    except Exception as e:
                        logger.warning(f"Video recording failed: {e}")
                
                # 6. Create next generation
                self.create_next_generation()
                
                # 7. Save checkpoint
                if generation % self.config.save_frequency == 0:
                    try:
                        self.save_checkpoint(generation)
                    except Exception as e:
                        logger.warning(f"Checkpoint save failed: {e}")
                
                # Update best network
                self.best_network = self.population[0]
                
                # Log progress
                if generation % 10 == 0:
                    self._log_progress()
            
            logger.info("✅ Training complete!")
            self.save_final_model()
            return self.best_network
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Try to save what we have
            if self.best_network is not None:
                try:
                    self.save_final_model()
                    logger.info("💾 Saved partial results before exit")
                except:
                    pass
            raise e
    
    def _calculate_hybrid_scores(self):
        """Calculate hybrid scores combining all components"""
        for network in self.population:
            # Base neuroevolution fitness
            hybrid_score = network.fitness * self.config.neuroevolution_weight
            
            # Add PG component if available
            if hasattr(network, 'pg_score'):
                hybrid_score += network.pg_score * self.config.pg_weight
            
            # Add LLM bonus for strategy alignment
            if hasattr(network, 'strategy_alignment'):
                hybrid_score += network.strategy_alignment * self.config.llm_weight * 100
            
            network.hybrid_score = hybrid_score
        
        # Re-sort by hybrid score
        self.population.sort(key=lambda x: getattr(x, 'hybrid_score', x.fitness), reverse=True)
        best_hybrid = getattr(self.population[0], 'hybrid_score', self.population[0].fitness)
        self.hybrid_score_history.append(best_hybrid)
    
    def _log_progress(self):
        """Log training progress"""
        recent_fitness = self.fitness_history[-10:] if len(self.fitness_history) > 10 else self.fitness_history
        recent_hybrid = self.hybrid_score_history[-10:] if len(self.hybrid_score_history) > 10 else self.hybrid_score_history
        
        logger.info("\n📊 Progress Report:")
        logger.info(f"  Avg Fitness (recent): {np.mean(recent_fitness):.1f}")
        logger.info(f"  Best Fitness (all-time): {max(self.fitness_history):.1f}")
        if recent_hybrid:
            logger.info(f"  Avg Hybrid Score (recent): {np.mean(recent_hybrid):.1f}")
        
        if self.pg_reward_history:
            logger.info(f"  Avg PG Reward (recent): {np.mean(self.pg_reward_history[-10:]):.1f}")
    
    def save_checkpoint(self, generation):
        """Save training checkpoint"""
        # Ensure we have a best network before saving
        if self.best_network is None:
            logger.warning("No best network available for checkpoint")
            return
            
        checkpoint = {
            'generation': generation,
            'best_network_state': self.best_network.state_dict(),
            'fitness_history': self.fitness_history,
            'hybrid_score_history': self.hybrid_score_history,
            'pg_reward_history': self.pg_reward_history,
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_gen_{generation:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        # Ensure we have a best network before saving
        if self.best_network is None:
            logger.warning("No best network available for final save")
            return
            
        final_model = {
            'network_state': self.best_network.state_dict(),
            'generation': self.generation,
            'best_fitness': max(self.fitness_history) if self.fitness_history else 0,
            'config': self.config
        }
        
        model_path = self.save_dir / "best_hybrid_model.pt"
        torch.save(final_model, model_path)
        logger.info(f"🏆 Saved final model: {model_path}")
        
        # Plot results
        self.plot_results()
    
    def plot_results(self):
        """Plot training results"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Fitness over time
        plt.subplot(1, 3, 1)
        plt.plot(self.fitness_history, label='Neuroevolution Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Neuroevolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Hybrid scores
        if self.hybrid_score_history:
            plt.subplot(1, 3, 2)
            plt.plot(self.hybrid_score_history, label='Hybrid Score', color='green')
            plt.xlabel('Generation')
            plt.ylabel('Hybrid Score')
            plt.title('Hybrid Score Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 3: PG rewards
        if self.pg_reward_history:
            plt.subplot(1, 3, 3)
            plt.plot(self.pg_reward_history, label='PG Reward', color='orange')
            plt.xlabel('PG Updates')
            plt.ylabel('Average Reward')
            plt.title('Policy Gradient Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"📈 Saved progress plot: {plot_path}")


def run_colab_demo():
    """Demo function optimized for Google Colab with high-quality video"""
    print("🚀 Hybrid Asteroids AI - Colab Demo")
    print("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # Install required packages (for Colab)
    print("\nInstalling requirements...")
    import subprocess
    import sys
    
    # Install packages
    packages = [
        "gymnasium[atari,accept-rom-license]",
        "ale-py",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "opencv-python"  # Added for video recording
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
    
    print("✅ Packages installation attempted")
    
    # Configure for demo with high-quality video
    config = HybridConfig(
        population_size=20,          # Smaller for demo
        generations=50,              # Shorter training
        episodes_per_eval=2,         # Faster evaluation
        use_llm_guidance=True,       # Enable LLM
        llm_model_name="microsoft/phi-2",  # Small model
        use_policy_gradient=True,    # Enable PG
        pg_episodes_per_update=3,    # Quick updates
        save_frequency=10,
        # High quality video settings
        video_frequency=5,           # Record every 5 generations
        video_episodes=3,            # 3 episodes per recording
        video_resolution=210,        # Full Atari resolution
        video_fps=60,               # 60 FPS for smooth playback
        record_frame_skip=1,        # NO frame skipping
        device="auto"
    )
    
    # Create trainer
    trainer = HybridTrainer(config)
    
    # Train
    print("\n🎮 Starting hybrid training with high-quality video recording...")
    print("📹 Videos will be recorded every 5 generations at 60 FPS")
    best_network = trainer.train_hybrid()
    
    print("\n✅ Training complete!")
    print(f"Best fitness achieved: {max(trainer.fitness_history):.1f}")
    
    # Test the best model with video recording
    print("\n🎬 Recording final gameplay video...")
    test_trained_model(
        str(trainer.save_dir / "best_hybrid_model.pt"),
        episodes=1,
        render=False,
        record_video=True
    )
    
    # Display video locations
    print("\n📁 Video files saved in:")
    video_dirs = sorted(trainer.save_dir.glob("videos_gen_*"))
    for video_dir in video_dirs:
        print(f"  - {video_dir}")
        videos = list(video_dir.glob("*.mp4"))
        for video in videos:
            print(f"    • {video.name}")
    
    return trainer, best_network


def test_trained_model(model_path: str, episodes: int = 5, render: bool = True, record_video: bool = True):
    """Test a trained hybrid model with optional video recording"""
    print(f"🎮 Testing trained model: {model_path}")
    
    # Load model with weights_only=False for custom classes
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create network
    network = HybridAsteroidsNetwork(config)
    network.load_state_dict(checkpoint['network_state'])
    network.eval()
    
    # Create environment
    if record_video:
        try:
            import cv2
        except ImportError:
            print("OpenCV not available for video recording. Install with: pip install opencv-python")
            record_video = False
    
    if record_video:
        # High quality video recording environment
        env = gym.make(
            config.env_id, 
            render_mode="rgb_array",
            frameskip=1,  # Disable base frameskip
            repeat_action_probability=0.0
        )
        
        # Setup video writer
        video_path = f"test_gameplay_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        
        # Minimal preprocessing for video
        env = AtariPreprocessing(
            env,
            noop_max=0,
            frame_skip=1,  # NO additional frame skipping
            screen_size=210,  # Full resolution
            terminal_on_life_loss=False,
            grayscale_obs=False,  # Keep RGB
            scale_obs=False
        )
    else:
        # Standard environment for testing
        env = gym.make(config.env_id, render_mode="human" if render else None, frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    
    # Test episodes
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        if record_video:
            # Initialize video writer with first frame
            if video_writer is None and episode == 0:
                h, w = state.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, 60, (w, h))
            
            # Grayscale frames for network
            gray_frames = deque(maxlen=config.frame_stack)
            gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            gray_state = cv2.resize(gray_state, (config.screen_size, config.screen_size))
            for _ in range(config.frame_stack):
                gray_frames.append(gray_state)
        else:
            frames = [state] * config.frame_stack
        
        while True:
            if record_video:
                # Write RGB frame to video
                if video_writer is not None:
                    video_writer.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
                
                # Prepare network input
                network_input = np.array(gray_frames)
                action = network.get_action(network_input, deterministic=True)
                
                # Step
                state, reward, terminated, truncated, _ = env.step(action)
                
                # Update grayscale frames
                gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                gray_state = cv2.resize(gray_state, (config.screen_size, config.screen_size))
                gray_frames.append(gray_state)
            else:
                # Standard testing
                stacked_state = np.array(frames)
                action = network.get_action(stacked_state, deterministic=True)
                
                state, reward, terminated, truncated, _ = env.step(action)
                
                frames.append(state)
                frames.pop(0)
            
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    # Cleanup
    if record_video and video_writer is not None:
        video_writer.release()
        print(f"\n🎬 Video saved: {video_path}")
    
    env.close()
    
    print(f"\nAverage reward: {np.mean(episode_rewards):.1f}")
    print(f"Best reward: {max(episode_rewards)}")
    
    return episode_rewards


def enhance_existing_network(existing_network_path: str, config: HybridConfig):
    """Enhance your existing neuroevolution network with hybrid capabilities"""
    print("🔧 Enhancing existing network with hybrid capabilities...")
    
    # Load existing network
    existing_state = torch.load(existing_network_path, map_location='cpu')
    
    # Create hybrid network
    hybrid_network = HybridAsteroidsNetwork(config)
    
    # Copy compatible weights from existing network
    existing_keys = set(existing_state.keys())
    hybrid_keys = set(hybrid_network.state_dict().keys())
    
    # Find matching keys
    compatible_keys = existing_keys.intersection(hybrid_keys)
    
    # Copy weights
    hybrid_state = hybrid_network.state_dict()
    for key in compatible_keys:
        if existing_state[key].shape == hybrid_state[key].shape:
            hybrid_state[key] = existing_state[key]
    
    hybrid_network.load_state_dict(hybrid_state)
    
    print(f"✅ Transferred {len(compatible_keys)} layers from existing network")
    print("🎯 New hybrid features added:")
    print("   - Policy gradient capabilities")
    print("   - Value function head")
    print("   - LLM strategy embedding")
    
    return hybrid_network


def create_showcase_video(model_path: str, output_path: str = "showcase.mp4"):
    """Create a high-quality showcase video of the trained AI"""
    print("🎬 Creating showcase video...")
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        return None
    
    # Load model with weights_only=False for custom classes
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create network
    network = HybridAsteroidsNetwork(config)
    network.load_state_dict(checkpoint['network_state'])
    network.eval()
    
    # Create HIGH QUALITY environment
    env = gym.make(
        config.env_id,
        render_mode="rgb_array",
        frameskip=1,  # Disable base frameskip
        repeat_action_probability=0.0
    )
    
    # Minimal preprocessing for maximum quality
    env = AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=1,  # NO additional frame skipping
        screen_size=210,  # Full resolution
        terminal_on_life_loss=False,
        grayscale_obs=False,  # Keep RGB
        scale_obs=False
    )
    
    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    # Play multiple episodes for showcase
    total_frames = 0
    best_score = 0
    
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        
        # Initialize video writer
        if video_writer is None:
            h, w = state.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, 60, (w, h))
            
            # Add title screen
            title_screen = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(title_screen, "HYBRID ASTEROIDS AI", 
                       (w//4, h//3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(title_screen, "Neuroevolution + LLM + Policy Gradients", 
                       (w//6, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Show title for 3 seconds
            for _ in range(60 * 3):
                video_writer.write(title_screen)
        
        # Grayscale frames for network
        gray_frames = deque(maxlen=config.frame_stack)
        gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        gray_state = cv2.resize(gray_state, (config.screen_size, config.screen_size))
        for _ in range(config.frame_stack):
            gray_frames.append(gray_state)
        
        while total_frames < 60 * 60:  # Max 60 seconds total
            # Add score overlay
            frame = state.copy()
            cv2.putText(frame, f"Score: {int(episode_reward)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Write frame
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Get action
            network_input = np.array(gray_frames)
            action = network.get_action(network_input, deterministic=True)
            
            # Step
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            total_frames += 1
            
            # Update grayscale frames
            gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            gray_state = cv2.resize(gray_state, (config.screen_size, config.screen_size))
            gray_frames.append(gray_state)
            
            if terminated or truncated:
                best_score = max(best_score, episode_reward)
                if episode < 2:  # More episodes
                    # Transition screen
                    transition = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(transition, f"Episode Score: {int(episode_reward)}", 
                               (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    for _ in range(60):  # 1 second transition
                        video_writer.write(transition)
                break
    
    # End screen
    end_screen = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(end_screen, f"Best Score: {int(best_score)}", 
               (w//3, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    for _ in range(60 * 2):  # 2 seconds
        video_writer.write(end_screen)
    
    video_writer.release()
    env.close()
    
    print(f"✅ Showcase video saved: {output_path}")
    print(f"🏆 Best score in showcase: {int(best_score)}")
    
    return output_path

"""
if __name__ == "__main__":
    # Run demo
    import sys
    
    if 'google.colab' in sys.modules:
        # Running in Colab
        trainer, best_network = run_colab_demo()
        
        # Create final showcase video
        create_showcase_video(
            "hybrid_asteroids_ai/best_hybrid_model.pt",
            "final_showcase.mp4"
        )
        
        # Mount drive to save videos
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Copy videos to drive
            import shutil
            drive_dir = "/content/drive/MyDrive/hybrid_asteroids_videos"
            os.makedirs(drive_dir, exist_ok=True)
            
            for video_dir in Path("hybrid_asteroids_ai").glob("videos_gen_*"):
                shutil.copytree(video_dir, f"{drive_dir}/{video_dir.name}", dirs_exist_ok=True)
            
            shutil.copy("final_showcase.mp4", drive_dir)
            print(f"\n📁 Videos copied to Google Drive: {drive_dir}")
        except Exception as e:
            print(f"\n⚠️ Could not mount Google Drive: {e}. Videos saved locally.")
    else:
        # Local run
        config = HybridConfig(
            population_size=40,
            generations=200,
            use_llm_guidance=True,
            use_policy_gradient=True,
            video_frequency=10,  # Record every 10 generations
            video_episodes=3,
            video_resolution=210,
            video_fps=60,
            device="auto"
        )
        
        trainer = HybridTrainer(config)
        best_network = trainer.train_hybrid()
        
        # Create showcase video
        create_showcase_video(
            "hybrid_asteroids_ai/best_hybrid_model.pt",
            "asteroids_ai_showcase.mp4"
        )
        
        # Test the best model
        test_trained_model(
            "hybrid_asteroids_ai/best_hybrid_model.pt",
            episodes=3,
            record_video=True
        )
"""