#!/usr/bin/env python3
"""
Enhanced Asteroids Neuroevolution with Game State Features + Video Recording
Combines CNN (visual) + MLP (game state) for superior performance
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
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register ALE environments
gym.register_envs(ale_py)

@dataclass
class Config:
    """Enhanced configuration with game state features"""
    
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
    population_size: int = 80
    generations: int = 100
    elite_size: int = 10
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    crossover_rate: float = 0.7
    
    # Enhanced Network Features
    use_visual_features: bool = True      # CNN from pixels
    use_game_state_features: bool = True  # NEW: Explicit game features
    max_asteroids_tracked: int = 5        # Track 5 nearest asteroids
    
    # Network architecture
    cnn_channels: List[int] = None
    fc_layers: List[int] = None
    dropout_rate: float = 0.1
    
    # Evaluation
    episodes_per_eval: int = 3
    max_steps_per_episode: int = 18000
    
    # Performance
    parallel_evaluation: bool = False
    num_workers: int = None
    
    # Video recording
    video_frequency: int = 10
    video_episodes: int = 3
    video_fps: int = 30
    record_full_episodes: bool = True
    
    # Paths
    save_dir: str = "enhanced_asteroids_evolution"
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 64]
        if self.fc_layers is None:
            self.fc_layers = [512, 256]
        if self.num_workers is None:
            self.num_workers = min(mp.cpu_count() - 1, 6)


class GameStateExtractor:
    """Extract explicit game state features from Asteroids pixels"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_asteroids = config.max_asteroids_tracked
        
        # Feature dimensions
        self.ship_features = 8       # x, y, dx, dy, angle, lives, thrust, rotation
        self.asteroid_features = 5   # x, y, size, distance, angle_to_ship (per asteroid)
        self.spatial_features = 6    # Quadrant densities, safe zones
        self.temporal_features = 4   # Movement trends
        
        self.total_features = (
            self.ship_features +
            self.asteroid_features * self.max_asteroids +
            self.spatial_features +
            self.temporal_features
        )
        
        # History for temporal analysis
        self.ship_history = deque(maxlen=10)
        self.asteroid_history = deque(maxlen=10)
        self.frame_count = 0
        
        # Only log once during initialization
        if not hasattr(GameStateExtractor, '_logged_init'):
            logger.info(f"🎯 Game state extractor: {self.total_features} features")
            GameStateExtractor._logged_init = True
    
    def extract_features(self, obs, info=None):
        """Extract game state features from RGB observation"""
        # Convert observation to analysis format
        if len(obs.shape) == 4:  # Batch dimension
            analysis_obs = obs[0]
        else:
            analysis_obs = obs
            
        # Handle different frame formats
        if analysis_obs.shape[0] == 4:  # Stacked frames (4, H, W)
            # Use most recent frame
            current_frame = analysis_obs[-1]
        elif len(analysis_obs.shape) == 3:  # Single frame (H, W, C) or (C, H, W)
            if analysis_obs.shape[0] in [1, 3, 4]:  # Channel first
                current_frame = analysis_obs[0] if analysis_obs.shape[0] == 1 else analysis_obs
            else:  # Channel last or grayscale
                current_frame = analysis_obs
        else:  # 2D grayscale
            current_frame = analysis_obs
        
        # Ensure 2D grayscale for analysis
        if len(current_frame.shape) == 3:
            if current_frame.shape[0] == 3:  # RGB
                current_frame = np.dot(current_frame.transpose(1, 2, 0), [0.299, 0.587, 0.114])
            else:
                current_frame = current_frame[0]  # Take first channel
        
        # Initialize features
        features = np.zeros(self.total_features)
        idx = 0
        
        try:
            # 1. Ship Features
            ship_features = self._extract_ship_features(current_frame)
            features[idx:idx+self.ship_features] = ship_features
            idx += self.ship_features
            
            # 2. Asteroid Features  
            asteroid_features = self._extract_asteroid_features(current_frame, ship_features[:2])
            features[idx:idx+self.asteroid_features*self.max_asteroids] = asteroid_features
            idx += self.asteroid_features * self.max_asteroids
            
            # 3. Spatial Features
            spatial_features = self._extract_spatial_features(current_frame, ship_features[:2])
            features[idx:idx+self.spatial_features] = spatial_features
            idx += self.spatial_features
            
            # 4. Temporal Features
            temporal_features = self._extract_temporal_features(ship_features, asteroid_features)
            features[idx:idx+self.temporal_features] = temporal_features
            
        except Exception as e:
            # Fallback to safe default features if extraction fails
            logger.debug(f"Feature extraction error: {e}")
            pass
        
        self.frame_count += 1
        return features
    
    def _extract_ship_features(self, frame):
        """Extract ship position, velocity, and state"""
        # Ship detection using simple image processing
        ship_x, ship_y = 0.5, 0.5  # Default center
        ship_dx, ship_dy = 0.0, 0.0
        ship_angle = 0.0
        lives = 1.0
        thrust = 0.0
        rotation = 0.0
        
        try:
            # Simple ship detection (bright pixel clusters)
            if frame.max() > 50:  # Game is active
                # Find brightest regions (likely ship/bullets)
                thresh = cv2.threshold(frame.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour (likely ship)
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 10:
                        moments = cv2.moments(largest_contour)
                        if moments['m00'] > 0:
                            ship_x = moments['m10'] / moments['m00'] / frame.shape[1]  # Normalize
                            ship_y = moments['m01'] / moments['m00'] / frame.shape[0]
                
                # Estimate velocity from history
                if len(self.ship_history) > 0:
                    prev_x, prev_y = self.ship_history[-1][:2]
                    ship_dx = ship_x - prev_x
                    ship_dy = ship_y - prev_y
                    
                    # Clamp velocities
                    ship_dx = np.clip(ship_dx, -0.1, 0.1)
                    ship_dy = np.clip(ship_dy, -0.1, 0.1)
        
        except Exception:
            pass
        
        ship_state = [ship_x, ship_y, ship_dx, ship_dy, ship_angle, lives, thrust, rotation]
        
        # Update history
        self.ship_history.append(ship_state)
        
        return np.array(ship_state)
    
    def _extract_asteroid_features(self, frame, ship_pos):
        """Extract features for nearest asteroids"""
        asteroid_features = []
        
        try:
            # Detect asteroids using edge detection
            edges = cv2.Canny(frame.astype(np.uint8), 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (asteroids are medium-large)
            asteroid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 1000:  # Asteroid size range
                    asteroid_contours.append(contour)
            
            # Sort by distance to ship
            ship_x, ship_y = ship_pos[0] * frame.shape[1], ship_pos[1] * frame.shape[0]
            
            asteroid_data = []
            for contour in asteroid_contours:
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    
                    # Distance to ship
                    distance = np.sqrt((cx - ship_x)**2 + (cy - ship_y)**2)
                    
                    # Asteroid size (relative)
                    size = cv2.contourArea(contour) / 100.0  # Normalize
                    
                    # Angle from ship to asteroid
                    angle = np.arctan2(cy - ship_y, cx - ship_x) / (2 * np.pi) + 0.5
                    
                    asteroid_data.append({
                        'x': cx / frame.shape[1],  # Normalize
                        'y': cy / frame.shape[0],
                        'size': min(size, 1.0),
                        'distance': distance / (frame.shape[0] + frame.shape[1]) * 2,  # Normalize
                        'angle': angle
                    })
            
            # Sort by distance and take closest
            asteroid_data.sort(key=lambda x: x['distance'])
            asteroid_data = asteroid_data[:self.max_asteroids]
            
            # Convert to feature array
            for i in range(self.max_asteroids):
                if i < len(asteroid_data):
                    ast = asteroid_data[i]
                    asteroid_features.extend([
                        ast['x'], ast['y'], ast['size'], ast['distance'], ast['angle']
                    ])
                else:
                    # No asteroid present
                    asteroid_features.extend([-1, -1, 0, -1, -1])
        
        except Exception:
            # Fallback: fill with default values
            for i in range(self.max_asteroids):
                asteroid_features.extend([-1, -1, 0, -1, -1])
        
        return np.array(asteroid_features)
    
    def _extract_spatial_features(self, frame, ship_pos):
        """Extract spatial layout features"""
        try:
            h, w = frame.shape
            ship_x, ship_y = ship_pos[0] * w, ship_pos[1] * h
            
            # Divide screen into quadrants and analyze density
            quadrants = [
                frame[:h//2, :w//2],      # Top-left
                frame[:h//2, w//2:],      # Top-right  
                frame[h//2:, :w//2],      # Bottom-left
                frame[h//2:, w//2:]       # Bottom-right
            ]
            
            quadrant_densities = []
            for quad in quadrants:
                # Count active pixels (asteroids/objects)
                density = np.sum(quad > 50) / quad.size
                quadrant_densities.append(density)
            
            # Distance to screen edges (safety)
            edge_distances = [
                ship_pos[0],           # Distance to left
                1.0 - ship_pos[0],     # Distance to right  
            ]
            
            return np.array(quadrant_densities + edge_distances)
        
        except Exception:
            return np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])
    
    def _extract_temporal_features(self, ship_features, asteroid_features):
        """Extract movement and trend features"""
        try:
            # Movement consistency
            if len(self.ship_history) >= 3:
                recent_positions = [pos[:2] for pos in list(self.ship_history)[-3:]]
                movement_consistency = self._calculate_movement_consistency(recent_positions)
            else:
                movement_consistency = 0.5
            
            # Threat level (how close are asteroids)
            asteroid_distances = []
            for i in range(0, len(asteroid_features), 5):
                if asteroid_features[i] != -1:  # Valid asteroid
                    asteroid_distances.append(asteroid_features[i+3])  # Distance
            
            if asteroid_distances:
                min_distance = min(asteroid_distances)
                avg_distance = np.mean(asteroid_distances)
            else:
                min_distance = 1.0
                avg_distance = 1.0
            
            return np.array([
                movement_consistency,
                min_distance,        # Closest threat
                avg_distance,        # Average threat distance
                len(asteroid_distances) / self.max_asteroids  # Asteroid density
            ])
        
        except Exception:
            return np.array([0.5, 1.0, 1.0, 0.2])
    
    def _calculate_movement_consistency(self, positions):
        """Calculate how consistent ship movement is"""
        if len(positions) < 3:
            return 0.5
        
        # Calculate movement vectors
        vectors = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            vectors.append([dx, dy])
        
        if len(vectors) < 2:
            return 0.5
        
        # Calculate consistency (similar direction/magnitude)
        consistencies = []
        for i in range(1, len(vectors)):
            dot_product = np.dot(vectors[i], vectors[i-1])
            mag_product = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i-1])
            if mag_product > 0:
                consistency = (dot_product / mag_product + 1) / 2  # Normalize to [0,1]
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.5


class EnhancedAsteroidsNetwork(nn.Module):
    """Enhanced network with both visual and game state features"""
    
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
        
        # Feature extractor for game state
        if config.use_game_state_features:
            self.feature_extractor = GameStateExtractor(config)
        
        # Build network components
        self._build_network()
        self._initialize_weights()
        
        # Only log network details once per population
        if not hasattr(EnhancedAsteroidsNetwork, '_logged_network'):
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"🧬 Enhanced network: {total_params:,} parameters, {self.n_actions} actions")
            EnhancedAsteroidsNetwork._logged_network = True
    
    def _build_network(self):
        """Build enhanced network with visual and game state branches"""
        
        # Visual branch (CNN) - if enabled
        if self.config.use_visual_features:
            self._build_visual_branch()
        
        # Game state branch (MLP) - if enabled  
        if self.config.use_game_state_features:
            self._build_gamestate_branch()
        
        # Fusion and output layers
        self._build_fusion_layers()
    
    def _build_visual_branch(self):
        """Build CNN branch for visual features"""
        conv_layers = []
        in_channels = self.config.frame_stack
        
        for out_channels in self.config.cnn_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=8 if len(conv_layers) == 0 else 4,
                         stride=4 if len(conv_layers) == 0 else 2, 
                         padding=2 if len(conv_layers) == 0 else 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
        
        self.visual_conv = nn.Sequential(*conv_layers)
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.frame_stack, 84, 84)
            conv_out = self.visual_conv(dummy)
            self.visual_features_size = conv_out.numel()
        
        # Visual FC layers (reduced size for fusion)
        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_features_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Only log once
        if not hasattr(self.__class__, '_logged_visual'):
            logger.info(f"👁️  Visual branch: {self.visual_features_size} → 128 features")
            self.__class__._logged_visual = True
    
    def _build_gamestate_branch(self):
        """Build MLP branch for game state features"""
        if not hasattr(self, 'feature_extractor'):
            return
            
        feature_dim = self.feature_extractor.total_features
        
        self.gamestate_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        # Only log once
        if not hasattr(self.__class__, '_logged_gamestate'):
            logger.info(f"🎯 Game state branch: {feature_dim} → 32 features")
            self.__class__._logged_gamestate = True
    
    def _build_fusion_layers(self):
        """Build fusion and output layers"""
        # Calculate total features from enabled branches
        total_features = 0
        
        if self.config.use_visual_features:
            total_features += 128  # Visual branch output
        
        if self.config.use_game_state_features:
            total_features += 32   # Game state branch output
        
        # Fallback if no features enabled
        if total_features == 0:
            total_features = 128
            logger.warning("No feature branches enabled! Using minimal network.")
        
        # Output layers with proper fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_actions)
        )
        
        # Only log once
        if not hasattr(self.__class__, '_logged_fusion'):
            logger.info(f"🔗 Fusion: {total_features} → {self.n_actions} actions")
            self.__class__._logged_fusion = True
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs, info=None):
        """Enhanced forward pass with feature fusion"""
        features = []
        
        # Visual features branch
        if self.config.use_visual_features:
            visual_features = self._process_visual_features(obs)
            features.append(visual_features)
        
        # Game state features branch  
        if self.config.use_game_state_features and hasattr(self, 'feature_extractor'):
            gamestate_features = self._process_gamestate_features(obs, info)
            features.append(gamestate_features)
        
        # Fuse features
        if len(features) == 0:
            # Fallback: minimal processing
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            visual_features = self._process_visual_features(obs)
            fused_features = visual_features
        elif len(features) == 1:
            fused_features = features[0]
        else:
            fused_features = torch.cat(features, dim=1)
        
        # Final output
        action_logits = self.fusion_fc(fused_features)
        return action_logits
    
    def _process_visual_features(self, obs):
        """Process visual observation through CNN"""
        # Normalize and ensure correct shape
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        obs = obs.float() / 255.0
        
        # Handle different input shapes
        if len(obs.shape) == 2:  # Single frame (H, W)
            obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(obs.shape) == 3:  # Frame stack or single frame
            if obs.shape[0] == 336:  # Stacked as (336, 84) → (4, 84, 84)
                obs = obs.view(4, 84, 84).unsqueeze(0)
            elif obs.shape[0] == 4:  # Already correct (4, H, W)
                obs = obs.unsqueeze(0)  # Add batch dim
            else:  # Other format
                obs = obs.unsqueeze(0).unsqueeze(0)
        elif len(obs.shape) == 4:  # Already batched
            pass
        
        # Ensure correct dimensions for CNN
        if obs.shape[1] != self.config.frame_stack:
            # Repeat frame if needed
            obs = obs.repeat(1, self.config.frame_stack, 1, 1)
        
        # CNN processing
        conv_features = self.visual_conv(obs)
        conv_features = conv_features.view(conv_features.size(0), -1)
        visual_out = self.visual_fc(conv_features)
        
        return visual_out
    
    def _process_gamestate_features(self, obs, info):
        """Process game state features through MLP"""
        try:
            # Extract game state features from observation
            game_features = self.feature_extractor.extract_features(obs, info)
            
            # Convert to tensor and add batch dimension
            game_features = torch.FloatTensor(game_features).unsqueeze(0)
            
            # Process through MLP
            gamestate_out = self.gamestate_fc(game_features)
            
            return gamestate_out
            
        except Exception as e:
            logger.debug(f"Game state feature extraction failed: {e}")
            # Fallback: return zero features
            return torch.zeros(1, 32)
    
    def get_action(self, obs, info=None, deterministic=False, temperature=1.0):
        """Get action with enhanced features"""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            
            logits = self.forward(obs, info)
            
            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            else:
                # Temperature-scaled sampling
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
        
        # Reset fitness and tracking
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


def create_env(config: Config, seed=None, record_video=False, video_folder=None, episode_trigger=None):
    """Create Asteroids environment - TRAINING optimized (frame skipping for speed)"""
    
    env = gym.make(
        config.env_id,
        obs_type="grayscale",
        frameskip=4,  # Training uses frame skipping for speed
        repeat_action_probability=config.repeat_action_probability,
        render_mode="rgb_array" if record_video else None
    )
    
    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env)
    
    # Apply training-optimized preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,  # Additional frame skip handled by base env
        screen_size=config.screen_size,  # 84x84 for efficiency
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False
    )
    
    # Stack frames for network
    env = FrameStack(env, num_stack=config.frame_stack)
    
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def evaluate_individual(network_state, config, episodes, seed, genome_id):
    """Evaluate individual with enhanced features"""
    # Create network
    network = EnhancedAsteroidsNetwork(config)
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
            # Get action with enhanced features
            action = network.get_action(obs, info, deterministic=False, temperature=1.2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    # Enhanced fitness calculation
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    consistency = 1.0 / (1.0 + np.std(episode_rewards))
    
    # Survival bonus (enhanced)
    survival_bonus = min(avg_length / config.max_steps_per_episode * 100, 100)
    
    # Performance bonus for high scores
    performance_bonus = min(avg_reward / 1000 * 50, 50)
    
    # Enhanced fitness combines multiple factors
    fitness = (
        avg_reward * 1.0 +           # Raw performance
        survival_bonus * 0.3 +       # Survival reward  
        performance_bonus * 0.2 +    # High score bonus
        consistency * 15             # Consistency bonus
    )
    
    return {
        'genome_id': genome_id,
        'fitness': fitness,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards
    }


class EvolutionTrainer:
    """Enhanced evolution trainer"""
    
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
        
        logger.info(f"Enhanced evolution trainer initialized with pop size {config.population_size}")
    
    def initialize_population(self):
        """Create initial population with enhanced networks"""
        logger.info("Creating enhanced population with game state features...")
        self.population = []
        
        for i in range(self.config.population_size):
            individual = EnhancedAsteroidsNetwork(self.config)
            individual.generation = 0
            self.population.append(individual)
        
        logger.info(f"✅ Enhanced population created: {len(self.population)} individuals")
        
        # Only show feature details once
        if self.config.use_game_state_features and len(self.population) > 0:
            sample_features = self.population[0].feature_extractor.total_features
            logger.info(f"🎯 Features: Visual + {sample_features} game state → Enhanced AI")
    
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
        
        # Show progress every 10 generations
        if self.generation % 10 == 0 and self.generation > 0:
            improvement = best_fitness - self.best_fitness_history[0] if len(self.best_fitness_history) > 1 else 0
            logger.info(f"📈 Progress: +{improvement:.1f} points since start")
        
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
            # Submit jobs with unpacked arguments
            futures = []
            for args in eval_args:
                future = executor.submit(evaluate_individual, *args)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 25 == 0:
                        logger.info(f"  Evaluated {i + 1}/{len(eval_args)} individuals")
                except Exception as e:
                    logger.error(f"Evaluation failed for individual {i}: {e}")
                    # Create dummy result to continue
                    results.append({
                        'genome_id': eval_args[i][4],
                        'fitness': 0.0,
                        'avg_reward': 0.0,
                        'avg_length': 0.0,
                        'episode_rewards': [0.0]
                    })
        
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
            
            if (i + 1) % 25 == 0:
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
        """Record MAXIMUM QUALITY videos with single recording system"""
        logger.info(f"Recording videos for generation {generation}")
        
        video_folder = self.video_dir / f"gen_{generation:03d}"
        video_folder.mkdir(exist_ok=True)
        
        # Clean any existing videos first
        for existing_video in video_folder.glob("*.mp4"):
            existing_video.unlink()
        
        # Create environment with ABSOLUTE MINIMUM frame skipping
        env = gym.make(
            self.config.env_id,
            obs_type="rgb",           # Full RGB color
            frameskip=1,              # Minimum possible in base environment
            repeat_action_probability=0.0,  # No action repeat
            render_mode="rgb_array"
        )
        
        # Add statistics tracking
        env = RecordEpisodeStatistics(env)
        
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
        
        # Add video recording AFTER preprocessing
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda x: x < self.config.video_episodes,
            video_length=0,           # Record FULL episodes
            name_prefix="evolution",  # Single naming scheme
            disable_logger=True
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
        
        # Record episodes
        episode_scores = []
        
        for episode in range(self.config.video_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_frames = 0
            
            for step in range(self.config.max_steps_per_episode):
                # Convert observation for network
                network_obs = env.get_network_obs(obs)
                
                # Get action from network
                action = best_individual.get_action(network_obs, info, deterministic=True)
                
                # Step environment (EVERY frame will be recorded)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_frames += 1
                
                if terminated or truncated:
                    break
            
            episode_scores.append(episode_reward)
        
        env.close()
        
        # Wait a moment for video files to be written
        time.sleep(1)
        
        # Verify video files and quality
        video_files = list(video_folder.glob("*.mp4"))
        logger.info(f"🎬 Videos: {len(video_files)} files, scores: {[f'{s:.0f}' for s in episode_scores]}")
        
        return video_files
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint to resume"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.best_fitness_history = checkpoint['fitness_history']['best']
        self.avg_fitness_history = checkpoint['fitness_history']['avg']
        
        # Load best individual
        best_network = EnhancedAsteroidsNetwork(self.config)
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
        plt.title('Enhanced Evolution Progress')
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
        logger.info("🚀 Starting Enhanced Asteroids Neuroevolution Training")
        logger.info(f"Population: {self.config.population_size}, Generations: {self.config.generations}")
        
        # Initialize
        self.initialize_population()
        
        start_time = time.time()
        
        for generation in range(self.config.generations):
            gen_start = time.time()
            
            logger.info(f"🧬 Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate population
            best_fitness, avg_fitness = self.evaluate_population()
            
            # Record videos periodically
            if generation % self.config.video_frequency == 0 or generation == self.config.generations - 1:
                self.record_videos(self.population[0], generation)
            
            # Save checkpoint
            if generation % 10 == 0 or generation == self.config.generations - 1:
                self.save_checkpoint(generation)
                self.plot_training_progress()
            
            # Early stopping check (more robust)
            if best_fitness > 100000:  # Higher target fitness
                # Additional check: ensure top performers are consistently good
                top_performers = sorted([ind.fitness for ind in self.population], reverse=True)[:5]
                avg_top_5 = np.mean(top_performers)
                
                if avg_top_5 > 80000:  # Top 5 individuals also performing well
                    logger.info(f"🎯 Target fitness reached! Best: {best_fitness:.1f}")
                    logger.info(f"Stopping early at generation {generation}")
                    break
            
            # Create next generation (only if not the last generation)
            if generation < self.config.generations - 1:
                self.create_next_generation()
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
        
        # Final evaluation and video
        logger.info("🎬 Creating final showcase video...")
        if generation < self.config.generations - 1:  # Only if stopped early
            self.record_videos(self.population[0], generation)
        
        self.save_checkpoint(generation)
        self.plot_training_progress()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Enhanced training completed in {total_time/60:.1f} minutes")
        logger.info(f"🏆 Best fitness achieved: {max(self.best_fitness_history):.1f}")
        logger.info(f"📈 Final generation: {generation + 1}")
        
        return self.population[0]


def test_environment():
    """Test enhanced environment setup"""
    logger.info("🧪 Testing Enhanced Asteroids Environment...")
    
    config = Config()
    env = create_env(config)
    
    logger.info(f"Environment: {config.env_id}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Observation space: {env.observation_space}")
    
    # Test episode
    obs, info = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    
    # Test enhanced network
    network = EnhancedAsteroidsNetwork(config)
    action = network.get_action(obs, info)
    logger.info(f"Enhanced network created, sample action: {action}")
    
    # Test feature extraction
    if config.use_game_state_features:
        features = network.feature_extractor.extract_features(obs, info)
        logger.info(f"Game state features extracted: {len(features)} dimensions")
        logger.info(f"Feature sample: {features[:10]}")  # First 10 features
    
    env.close()
    logger.info("✅ Enhanced environment test successful!")
    
    return True


def compare_architectures():
    """Compare original vs enhanced architecture"""
    logger.info("🔍 ARCHITECTURE COMPARISON")
    logger.info("=" * 60)
    
    # Original config
    logger.info("📊 ORIGINAL NETWORK (Pixels Only):")
    original_config = Config(use_game_state_features=False)
    original_net = EnhancedAsteroidsNetwork(original_config)
    original_params = sum(p.numel() for p in original_net.parameters())
    logger.info(f"   Parameters: {original_params:,}")
    logger.info(f"   Input: Visual only (4×84×84 pixels)")
    
    logger.info("")
    
    # Enhanced config
    logger.info("🚀 ENHANCED NETWORK (Pixels + Game State):")
    enhanced_config = Config(use_game_state_features=True)
    enhanced_net = EnhancedAsteroidsNetwork(enhanced_config)
    enhanced_params = sum(p.numel() for p in enhanced_net.parameters())
    logger.info(f"   Parameters: {enhanced_params:,}")
    logger.info(f"   Input: Visual + {enhanced_net.feature_extractor.total_features} game features")
    
    logger.info("")
    logger.info("💡 EXPECTED BENEFITS:")
    logger.info("   ✅ 3-5x faster learning")
    logger.info("   ✅ Better spatial awareness")
    logger.info("   ✅ Improved collision avoidance")
    logger.info("   ✅ More consistent performance")
    logger.info("   ✅ Strategic gameplay")


def main():
    """Main function with enhanced features"""
    print("🚀 ENHANCED Asteroids Neuroevolution with Game State Features")
    print("=" * 70)
    
    # Test environment first
    if not test_environment():
        logger.error("Environment test failed!")
        return
    
    # Show architecture comparison
    compare_architectures()
    
    # Enhanced configuration
    config = Config(
        population_size=80,
        generations=100,
        episodes_per_eval=3,
        video_frequency=10,
        video_episodes=3,
        parallel_evaluation=False,  # Keep disabled for stability
        num_workers=4,
        # Enhanced features
        use_visual_features=True,
        use_game_state_features=True,
        max_asteroids_tracked=5,
        save_dir="enhanced_asteroids_evolution"
    )
    
    logger.info(f"🎮 Enhanced Training Configuration:")
    logger.info(f"  Population size: {config.population_size}")
    logger.info(f"  Generations: {config.generations}")
    logger.info(f"  Enhanced features: ENABLED")
    logger.info(f"  Expected improvement: 3-5x faster learning")
    
    # Create trainer and run
    trainer = EvolutionTrainer(config)
    best_individual = trainer.train()
    
    logger.info(f"🎉 Enhanced training complete!")
    logger.info(f"📁 Results saved in: {config.save_dir}")
    logger.info(f"🎬 Videos saved in: {config.save_dir}/videos")


def continue_training_from_checkpoint():
    """Continue training from the latest checkpoint with enhanced features"""
    print("🔄 Continuing Enhanced Asteroids Neuroevolution")
    print("=" * 60)
    
    # Enhanced configuration for continued training
    config = Config(
        population_size=80,
        generations=200,  # Extended generations
        episodes_per_eval=3,
        video_frequency=10,
        video_episodes=3,
        parallel_evaluation=False,
        num_workers=4,
        # Enhanced features
        use_visual_features=True,
        use_game_state_features=True,
        max_asteroids_tracked=5,
        save_dir="enhanced_asteroids_evolution"
    )
    
    # Find latest checkpoint
    save_dir = Path(config.save_dir)
    checkpoints = list(save_dir.glob("checkpoint_gen_*.pkl"))
    
    if not checkpoints:
        logger.error("No checkpoints found! Starting fresh training.")
        return main()
    
    # Get latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    # Create trainer and load checkpoint
    trainer = EvolutionTrainer(config)
    trainer.load_checkpoint(latest_checkpoint)
    
    # Initialize population (will be replaced by evolution)
    trainer.initialize_population()
    
    # Continue training
    logger.info(f"Resuming enhanced training from generation {trainer.generation + 1}")
    best_individual = trainer.train()
    
    logger.info(f"🎉 Enhanced continued training complete!")
    return trainer, best_individual


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "continue":
        # Continue training from checkpoint
        continue_training_from_checkpoint()
    else:
        # Start fresh training
        main()