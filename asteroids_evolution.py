#!/usr/bin/env python3
"""
Streamlined Asteroids Neuroevolution with Video Recording
Optimized for MacBook Pro with uv package manager
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, 
    RecordEpisodeStatistics,
    RecordVideo
)

# Try different FrameStack import paths for compatibility
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    try:
        from gymnasium.wrappers.frame_stack import FrameStack
    except ImportError:
        try:
            from gymnasium.experimental.wrappers import FrameStackObservation as FrameStack
        except ImportError:
            # Fallback: create our own simple frame stacking wrapper
            import gymnasium as gym
            from collections import deque
            
            class FrameStack(gym.Wrapper):
                def __init__(self, env, num_stack):
                    super().__init__(env)
                    self.num_stack = num_stack
                    self.frames = deque(maxlen=num_stack)
                    
                    # Create correct observation space for stacked frames
                    obs_space = env.observation_space
                    if len(obs_space.shape) == 2:  # (H, W)
                        new_shape = (num_stack, obs_space.shape[0], obs_space.shape[1])
                    else:  # Already has channels
                        new_shape = (num_stack * obs_space.shape[0],) + obs_space.shape[1:]
                    
                    self.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=new_shape, dtype=obs_space.dtype
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
                    # Stack frames along the first dimension: (num_stack, H, W)
                    return np.stack(list(self.frames), axis=0)
import random
import time
import uuid
import copy
from pathlib import Path
import pickle
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register ALE environments
gym.register_envs(ale_py)

@dataclass
class Config:
    """Configuration for Asteroids Neuroevolution"""
    
    # Environment
    env_id: str = "ALE/Asteroids-v5"
    render_mode: Optional[str] = None
    frameskip: int = 4
    repeat_action_probability: float = 0.0
    
    # Preprocessing
    screen_size: int = 84
    terminal_on_life_loss: bool = True
    frame_stack: int = 4
    
    # Evolution
    population_size: int = 100
    generations: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    crossover_rate: float = 0.7
    
    # Network
    cnn_channels: List[int] = None
    fc_layers: List[int] = None
    dropout_rate: float = 0.1
    
    # Evaluation
    episodes_per_eval: int = 3
    max_steps_per_episode: int = 18000
    
    # Performance
    parallel_evaluation: bool = True
    num_workers: int = None
    
    # Video recording
    video_frequency: int = 10  # Record every N generations
    video_episodes: int = 3    # Episodes to record
    video_quality: str = "high"  # "low", "medium", "high"
    video_fps: int = 30        # FPS for smoother videos
    record_full_episodes: bool = True  # Record complete episodes
    
    # Paths
    save_dir: str = "asteroids_evolution"
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 64]
        if self.fc_layers is None:
            self.fc_layers = [512, 256]
        if self.num_workers is None:
            # Conservative for MacBook Pro
            self.num_workers = min(mp.cpu_count() - 1, 6)


def create_env(config: Config, seed=None, record_video=False, video_folder=None, episode_trigger=None):
    """Create Asteroids environment with optional video recording"""
    
    env = gym.make(
        config.env_id,
        obs_type="grayscale",
        frameskip=config.frameskip,
        repeat_action_probability=config.repeat_action_probability,
        render_mode="rgb_array" if record_video else None
    )
    
    # Add statistics tracking
    env = RecordEpisodeStatistics(env)
    
    # Add video recording if requested
    if record_video and video_folder:
        if episode_trigger is None:
            episode_trigger = lambda x: True
        
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            disable_logger=True
        )
    
    # Apply preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,  # Already handled in base env
        screen_size=config.screen_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False
    )
    
    # Stack frames
    env = FrameStack(env, num_stack=config.frame_stack)
    
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


class AsteroidsNetwork(nn.Module):
    """CNN network for Asteroids"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.genome_id = str(uuid.uuid4())
        
        # Get action space size
        temp_env = create_env(config)
        self.n_actions = temp_env.action_space.n
        temp_env.close()
        
        # Fitness tracking
        self.fitness = 0.0
        self.episode_rewards = []
        self.generation = 0
        
        self._build_network()
        self._initialize_weights()
        
        logger.info(f"Created network with {self.n_actions} actions")
    
    def _build_network(self):
        """Build CNN architecture"""
        # CNN layers
        conv_layers = []
        in_channels = self.config.frame_stack
        
        for out_channels in self.config.cnn_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=8 if len(conv_layers) == 0 else 4, 
                         stride=4 if len(conv_layers) == 0 else 2, padding=2 if len(conv_layers) == 0 else 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.frame_stack, 84, 84)  # Correct shape for frame stack
            conv_out = self.conv(dummy)
            self.conv_output_size = conv_out.numel()
        
        # FC layers
        fc_layers = []
        prev_size = self.conv_output_size
        
        for hidden_size in self.config.fc_layers:
            fc_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_size = hidden_size
        
        fc_layers.append(nn.Linear(prev_size, self.n_actions))
        self.fc = nn.Sequential(*fc_layers)
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Normalize input
        x = x.float() / 255.0
        
        # Handle different input shapes from frame stacking
        if len(x.shape) == 2:  # Single frame: (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel: (1, 1, H, W)
        elif len(x.shape) == 3:  # Frame stack: (frames*H, W) or (frames, H, W)
            if x.shape[0] == 336:  # Stacked frames as (336, 84) -> (4, 84, 84)
                x = x.view(4, 84, 84)
            x = x.unsqueeze(0)  # Add batch dimension: (1, 4, 84, 84)
        elif len(x.shape) == 4:  # Already batched: (B, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Ensure we have the right shape for CNN: (B, C, H, W)
        if x.shape[2] != 84 or x.shape[3] != 84:
            # Reshape if needed - handle the stacked frames case
            if x.shape[1] == 336:  # (B, 336, 84) -> (B, 4, 84, 84)
                x = x.view(x.shape[0], 4, 84, 84)
        
        # CNN features
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc(x)
        return x
    
    def get_action(self, state, deterministic=False, temperature=1.0):
        """Get action from state"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            
            logits = self.forward(state)
            
            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            else:
                # Temperature scaling
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
                return action
    
    def mutate(self):
        """Mutate network weights"""
        with torch.no_grad():
            for param in self.parameters():
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * self.config.mutation_strength
                    param.add_(noise)
        
        # Reset fitness
        self.fitness = 0.0
        self.episode_rewards = []
    
    def crossover(self, other):
        """Create offspring via crossover"""
        offspring = copy.deepcopy(self)
        offspring.genome_id = str(uuid.uuid4())
        offspring.generation = max(self.generation, other.generation) + 1
        offspring.fitness = 0.0
        offspring.episode_rewards = []
        
        with torch.no_grad():
            for (off_param, p1_param, p2_param) in zip(
                offspring.parameters(), self.parameters(), other.parameters()
            ):
                if random.random() < self.config.crossover_rate:
                    # Uniform crossover
                    mask = torch.rand_like(off_param) < 0.5
                    off_param[mask] = p2_param[mask]
        
        return offspring
    
    def clone(self):
        """Create exact copy"""
        cloned = copy.deepcopy(self)
        cloned.genome_id = str(uuid.uuid4())
        return cloned


def evaluate_individual(network_state, config, episodes, seed, genome_id):
    """Evaluate individual in separate process"""
    # Create network
    network = AsteroidsNetwork(config)
    network.load_state_dict(network_state)
    network.eval()
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = create_env(config, seed=seed)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(config.max_steps_per_episode):
            action = network.get_action(obs, deterministic=False, temperature=1.2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    # Calculate fitness
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    consistency = 1.0 / (1.0 + np.std(episode_rewards))
    
    # Survival bonus
    survival_bonus = min(avg_length / config.max_steps_per_episode * 50, 50)
    
    fitness = avg_reward + survival_bonus * 0.2 + consistency * 10
    
    return {
        'genome_id': genome_id,
        'fitness': fitness,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards
    }


class EvolutionTrainer:
    """Main evolution trainer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Setup directories
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.video_dir = self.save_dir / "videos"
        self.video_dir.mkdir(exist_ok=True)
        
        logger.info(f"Evolution trainer initialized with pop size {config.population_size}")
    
    def initialize_population(self):
        """Create initial population"""
        logger.info("Creating initial population...")
        self.population = []
        
        for i in range(self.config.population_size):
            individual = AsteroidsNetwork(self.config)
            individual.generation = 0
            self.population.append(individual)
        
        logger.info(f"Created {len(self.population)} individuals")
    
    def evaluate_population(self):
        """Evaluate entire population"""
        logger.info(f"Evaluating generation {self.generation}")
        
        if self.config.parallel_evaluation:
            results = self._evaluate_parallel()
        else:
            results = self._evaluate_sequential()
        
        # Update fitness values
        for result in results:
            for individual in self.population:
                if individual.genome_id == result['genome_id']:
                    individual.fitness = result['fitness']
                    individual.episode_rewards = result['episode_rewards']
                    break
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track statistics
        fitnesses = [ind.fitness for ind in self.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        logger.info(f"Gen {self.generation}: Best={best_fitness:.1f}, Avg={avg_fitness:.1f}")
        
        return best_fitness, avg_fitness
    
    def _evaluate_parallel(self):
        """Parallel evaluation"""
        eval_args = []
        for individual in self.population:
            args = (
                individual.state_dict(),
                self.config,
                self.config.episodes_per_eval,
                random.randint(0, 100000),
                individual.genome_id
            )
            eval_args.append(args)
        
        results = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_args = {executor.submit(evaluate_individual, args): args for args in eval_args}
            
            for i, future in enumerate(as_completed(future_to_args)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Evaluated {i + 1}/{len(eval_args)} individuals")
        
        return results
    
    def _evaluate_sequential(self):
        """Sequential evaluation for debugging"""
        results = []
        for i, individual in enumerate(self.population):
            args = (
                individual.state_dict(),
                self.config,
                self.config.episodes_per_eval,
                random.randint(0, 100000),
                individual.genome_id
            )
            
            result = evaluate_individual(*args)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(self.population)} individuals")
        
        return results
    
    def create_next_generation(self):
        """Create next generation via selection and reproduction"""
        next_population = []
        
        # Elite selection
        elite_size = self.config.elite_size
        elites = self.population[:elite_size]
        
        for elite in elites:
            elite_copy = elite.clone()
            elite_copy.generation = self.generation + 1
            next_population.append(elite_copy)
        
        # Fill remaining spots with offspring
        while len(next_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring = parent1.crossover(parent2)
            else:
                offspring = parent1.clone()
            
            # Mutation
            offspring.mutate()
            offspring.generation = self.generation + 1
            
            next_population.append(offspring)
        
        self.population = next_population
        self.generation += 1
    
    def _tournament_selection(self, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(self.population[:50], min(tournament_size, 50))
        return max(tournament, key=lambda x: x.fitness)
    
    def record_videos(self, best_individual, generation):
        """Record MAXIMUM QUALITY videos with absolute minimum frame skipping"""
        logger.info(f"Recording MAXIMUM QUALITY videos for generation {generation}")
        
        video_folder = self.video_dir / f"gen_{generation:03d}"
        video_folder.mkdir(exist_ok=True)
        
        # Create environment with ABSOLUTE MINIMUM frame skipping
        env = gym.make(
            self.config.env_id,
            obs_type="rgb",           # Full RGB color
            frameskip=1,              # Minimum possible in base environment
            repeat_action_probability=0.0,  # No action repeat
            render_mode="rgb_array"
        )
        
        # Add statistics but NO preprocessing that would skip frames
        env = RecordEpisodeStatistics(env)
        
        # Configure RecordVideo for maximum quality
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda x: x < self.config.video_episodes,
            video_length=0,           # Record FULL episodes
            name_prefix="asteroids_evolution",
            disable_logger=True
        )
        
        # AtariPreprocessing with MINIMUM frame_skip (1 = no additional skipping)
        env = AtariPreprocessing(
            env,
            noop_max=0,               # No random no-ops at start
            frame_skip=1,             # MINIMUM = 1 (no additional frame skipping)
            screen_size=210,          # Higher resolution than training
            terminal_on_life_loss=self.config.terminal_on_life_loss,
            grayscale_obs=False,      # Keep full RGB color
            grayscale_newaxis=False,
            scale_obs=False           # Keep original pixel values
        )
        
        # Create a wrapper to convert RGB to format network expects
        class NetworkCompatibleWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                return obs, info
                
            def step(self, action):
                return self.env.step(action)
                
            def get_network_obs(self, rgb_obs):
                """Convert RGB observation to network-compatible format"""
                # Convert RGB to grayscale
                if len(rgb_obs.shape) == 3:
                    gray = np.dot(rgb_obs[...,:3], [0.2989, 0.5870, 0.1140])
                    # Resize to network expected size
                    gray_resized = cv2.resize(gray, (84, 84))
                    # Create fake frame stack (4 identical frames)
                    return np.stack([gray_resized] * 4, axis=0)
                return rgb_obs
        
        env = NetworkCompatibleWrapper(env)
        
        logger.info(f"MAXIMUM QUALITY video settings:")
        logger.info(f"  - Base environment frameskip: 1 (minimum)")
        logger.info(f"  - AtariPreprocessing frame_skip: 1 (minimum)")  
        logger.info(f"  - Effective frame skipping: NONE")
        logger.info(f"  - Resolution: 210x160 RGB")
        logger.info(f"  - Color: Full RGB (not grayscale)")
        logger.info(f"  - Frame rate: ~60 FPS")
        
        # Record episodes with detailed logging
        total_frames_recorded = 0
        for episode in range(self.config.video_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_frames = 0
            
            logger.info(f"Recording episode {episode + 1} with ZERO frame skipping...")
            
            for step in range(self.config.max_steps_per_episode):
                # Convert observation for network
                network_obs = env.get_network_obs(obs)
                
                # Get action from network
                action = best_individual.get_action(network_obs, deterministic=True)
                
                # Step environment (EVERY frame will be recorded)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_frames += 1
                
                # Log progress for long episodes
                if step > 0 and step % 2000 == 0:
                    logger.info(f"    Episode {episode + 1}: {step} steps, {episode_reward:.1f} points")
                
                if terminated or truncated:
                    break
            
            total_frames_recorded += episode_frames
            logger.info(f"  Episode {episode + 1}: {episode_reward:.1f} points, {episode_frames} frames recorded")
        
        env.close()
        
        # Verify video files and quality
        video_files = list(video_folder.glob("*.mp4"))
        logger.info(f"🎬 PRESENTATION-QUALITY VIDEOS CREATED:")
        logger.info(f"  - Files: {len(video_files)} videos")
        logger.info(f"  - Location: {video_folder}")
        logger.info(f"  - Total frames: {total_frames_recorded}")
        logger.info(f"  - Quality: MAXIMUM (no frame skipping)")
        
        for video_file in video_files:
            file_size = video_file.stat().st_size / (1024*1024)  # MB
            logger.info(f"  - {video_file.name}: {file_size:.1f} MB")
        
        logger.info(f"✅ Videos ready for colleague presentation!")
        
        return video_files
    
    def _create_summary_video(self, video_folder, generation):
        """Create summary video with stats overlay"""
        try:
            video_files = list(video_folder.glob("*.mp4"))
            if not video_files:
                return
            
            # For now, just log the video files
            logger.info(f"Created {len(video_files)} video files in {video_folder}")
            
        except Exception as e:
            logger.warning(f"Failed to create summary video: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint to resume"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.best_fitness_history = checkpoint['fitness_history']['best']
        self.avg_fitness_history = checkpoint['fitness_history']['avg']
        
        # Load best individual
        best_network = AsteroidsNetwork(self.config)
        best_network.load_state_dict(checkpoint['best_individual'])
        
        logger.info(f"✅ Checkpoint loaded:")
        logger.info(f"  - Generation: {self.generation}")
        logger.info(f"  - Best fitness: {checkpoint['best_fitness']}")
        logger.info(f"  - Ready to resume training")
        
        return best_network

    def save_checkpoint(self, generation):
        """Save training checkpoint"""
        checkpoint = {
            'generation': generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'best_individual': self.population[0].state_dict(),
            'fitness_history': {
                'best': self.best_fitness_history,
                'avg': self.avg_fitness_history
            },
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_gen_{generation:03d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save best model separately
        best_model_path = self.save_dir / "best_model.pt"
        torch.save(self.population[0].state_dict(), best_model_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        """Save training checkpoint"""
        checkpoint = {
            'generation': generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'best_individual': self.population[0].state_dict(),
            'fitness_history': {
                'best': self.best_fitness_history,
                'avg': self.avg_fitness_history
            },
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_gen_{generation:03d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save best model separately
        best_model_path = self.save_dir / "best_model.pt"
        torch.save(self.population[0].state_dict(), best_model_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.best_fitness_history:
            return
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='Best Fitness', linewidth=2)
        plt.plot(self.avg_fitness_history, label='Average Fitness', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        recent_gens = min(20, len(self.best_fitness_history))
        if recent_gens > 1:
            recent_best = self.best_fitness_history[-recent_gens:]
            recent_avg = self.avg_fitness_history[-recent_gens:]
            generations = list(range(len(self.best_fitness_history) - recent_gens, len(self.best_fitness_history)))
            
            plt.plot(generations, recent_best, 'o-', label='Best (Recent)', linewidth=2)
            plt.plot(generations, recent_avg, 'o-', label='Avg (Recent)', alpha=0.7)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Recent Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        progress_path = self.save_dir / f"training_progress_gen_{self.generation:03d}.png"
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved progress plot to {progress_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting Asteroids Neuroevolution Training")
        logger.info(f"Population: {self.config.population_size}, Generations: {self.config.generations}")
        
        # Initialize
        self.initialize_population()
        
        start_time = time.time()
        
        for generation in range(self.config.generations):
            gen_start = time.time()
            
            logger.info(f"🧬 Starting Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate population
            best_fitness, avg_fitness = self.evaluate_population()
            
            # Record videos periodically
            if generation % self.config.video_frequency == 0 or generation == self.config.generations - 1:
                self.record_videos(self.population[0], generation)
            
            # Save checkpoint
            if generation % 10 == 0 or generation == self.config.generations - 1:
                self.save_checkpoint(generation)
                self.plot_training_progress()
            
            # More robust early stopping check
            if best_fitness > 100000:  # High target fitness
                # Additional check: ensure top performers are consistently good
                top_performers = sorted([ind.fitness for ind in self.population], reverse=True)[:5]
                avg_top_5 = np.mean(top_performers)
                
                if avg_top_5 > 80000:  # Top 5 individuals also performing well
                    logger.info(f"🎯 Target fitness reached! Best: {best_fitness:.1f}")
                    logger.info(f"🏆 Top 5 average: {avg_top_5:.1f} (also excellent)")
                    logger.info(f"Stopping early at generation {generation}")
                    break
            
            # Create next generation (only if not the last generation)
            if generation < self.config.generations - 1:
                logger.info(f"Creating generation {generation + 2}...")
                self.create_next_generation()
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            
            logger.info(f"Generation {generation + 1} completed in {gen_time:.1f}s (Total: {total_time/60:.1f}m)")
            logger.info(f"Progress: {generation + 1}/{self.config.generations} generations ({((generation + 1)/self.config.generations)*100:.1f}%)")
        
        # Final evaluation and video
        logger.info("🎬 Creating final showcase video...")
        if generation < self.config.generations - 1:  # Only if stopped early
            self.record_videos(self.population[0], generation)
        
        self.save_checkpoint(generation)
        self.plot_training_progress()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time/60:.1f} minutes")
        logger.info(f"🏆 Best fitness achieved: {max(self.best_fitness_history):.1f}")
        logger.info(f"📈 Final generation: {generation + 1}")
        
        return self.population[0]


def test_environment():
    """Test environment setup"""
    logger.info("🧪 Testing Asteroids environment setup...")
    
    config = Config()
    env = create_env(config)
    
    logger.info(f"Environment: {config.env_id}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    
    # Test episode
    obs, info = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    
    # Test network
    network = AsteroidsNetwork(config)
    action = network.get_action(obs)
    logger.info(f"Network created, sample action: {action}")
    
    env.close()
    logger.info("✅ Environment test successful!")
    
    return True


def main():
    """Main function"""
    print("🎮 Asteroids Neuroevolution with Video Recording")
    print("=" * 60)
    
    # Test environment first
    if not test_environment():
        logger.error("Environment test failed!")
        return
    
    # Configuration
    config = Config(
        population_size=80,
        generations=100,  # Full training
        episodes_per_eval=3,
        video_frequency=10,  # Record every 10 generations
        video_episodes=3,
        parallel_evaluation=False,  # Disable for stability
        num_workers=4
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Population size: {config.population_size}")
    logger.info(f"  Generations: {config.generations}")
    logger.info(f"  Video every {config.video_frequency} generations")
    logger.info(f"  Parallel workers: {config.num_workers}")
    
    # Create trainer and run
    trainer = EvolutionTrainer(config)
    best_individual = trainer.train()
    
    logger.info(f"🎉 Training complete!")
    logger.info(f"📁 Results saved in: {config.save_dir}")
    logger.info(f"🎬 Videos saved in: {config.save_dir}/videos")


if __name__ == "__main__":
    main()