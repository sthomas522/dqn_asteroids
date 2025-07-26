#!/usr/bin/env python3
"""
Advanced Plateau-Breaking Strategies for Asteroids Neuroevolution
Drop-in replacement for your existing trainer with advanced techniques
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AdvancedConfig:
    """Enhanced configuration with all plateau-breaking features"""
    
    # Base configuration (keep your existing values)
    env_id: str = "ALE/Asteroids-v5"
    population_size: int = 80
    generations: int = 2000
    elite_size: int = 10
    episodes_per_eval: int = 5  # Increased for stability
    max_steps_per_episode: int = 18000
    
    # Plateau Detection
    plateau_detection_window: int = 15      # Generations to detect plateau
    min_improvement_threshold: float = 15.0  # Minimum fitness improvement
    plateau_severity_levels: int = 3        # Different intervention levels
    
    # Adaptive Mutation System
    base_mutation_rate: float = 0.15
    base_mutation_strength: float = 0.1
    plateau_mutation_multiplier: float = 2.5  # How much to increase during plateaus
    max_mutation_rate: float = 0.45
    max_mutation_strength: float = 0.25
    mutation_decay_rate: float = 0.95       # Decay after improvements
    
    # Dynamic Population Management  
    enable_population_growth: bool = True
    population_growth_factor: float = 1.15  # 15% growth during severe plateaus
    max_population_size: int = 120
    population_refresh_rate: float = 0.25   # Replace 25% worst performers
    
    # Diversity Preservation (Novelty Search)
    enable_novelty_search: bool = True
    novelty_weight: float = 0.3             # Weight for novelty in selection
    novelty_archive_size: int = 100
    behavioral_distance_threshold: float = 0.5
    k_nearest_neighbors: int = 15           # For novelty calculation
    
    # Multi-Objective Evolution
    enable_multi_objective: bool = True
    survival_weight: float = 0.4
    score_weight: float = 0.5
    consistency_weight: float = 0.1
    risk_taking_weight: float = 0.05        # Bonus for calculated risks
    
    # Speciation (Species-based Evolution)
    enable_speciation: bool = True
    compatibility_threshold: float = 3.0    # Genetic distance threshold
    species_elitism: int = 2                # Top individuals per species
    interspecies_mating_rate: float = 0.001 # Cross-species breeding rate
    
    # Curriculum Learning
    enable_curriculum: bool = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {
            "name": "basic_survival",
            "generations": 50,
            "max_steps": 8000,
            "target_fitness": 300,
            "focus": "survival_time"
        },
        {
            "name": "evasion_mastery", 
            "generations": 100,
            "max_steps": 12000,
            "target_fitness": 600,
            "focus": "collision_avoidance"
        },
        {
            "name": "scoring_focus",
            "generations": 150,
            "max_steps": 16000,
            "target_fitness": 1000,
            "focus": "score_optimization"
        },
        {
            "name": "mastery",
            "generations": 999999,  # Final stage
            "max_steps": 18000,
            "target_fitness": 2000,
            "focus": "overall_performance"
        }
    ])
    
    # Network Architecture Enhancements
    enable_residual_connections: bool = True
    enable_dropout_scheduling: bool = True
    base_dropout_rate: float = 0.1
    plateau_dropout_rate: float = 0.2       # Higher dropout during plateaus
    
    # Experience Memory (for behavioral analysis)
    enable_experience_memory: bool = True
    memory_size: int = 500
    elite_experience_weight: float = 2.0    # Weight elite experiences more
    
    # Advanced Selection Strategies
    tournament_size: int = 5
    enable_fitness_sharing: bool = True     # Reduce fitness for similar individuals
    enable_age_layered_evolution: bool = True  # Consider individual age in selection


class BehavioralDescriptor:
    """Create behavioral fingerprints for novelty search"""
    
    @staticmethod
    def extract_behavior(episode_data: Dict) -> np.ndarray:
        """Extract behavioral descriptor from episode data"""
        
        # Movement patterns (8 features)
        positions = episode_data.get('positions', [(0.5, 0.5)] * 10)
        positions = np.array(positions)
        
        movement_features = [
            np.std(positions[:, 0]),                    # X variance
            np.std(positions[:, 1]),                    # Y variance  
            np.mean(np.diff(positions[:, 0])),          # Avg X velocity
            np.mean(np.diff(positions[:, 1])),          # Avg Y velocity
            np.max(positions[:, 0]) - np.min(positions[:, 0]),  # X range
            np.max(positions[:, 1]) - np.min(positions[:, 1]),  # Y range
            len(np.where(np.diff(np.diff(positions[:, 0])) > 0.01)[0]),  # Direction changes X
            len(np.where(np.diff(np.diff(positions[:, 1])) > 0.01)[0])   # Direction changes Y
        ]
        
        # Action patterns (6 features)
        actions = episode_data.get('actions', [0] * 100)
        action_counts = np.bincount(actions, minlength=14)
        action_distribution = action_counts / len(actions)
        
        # Focus on key action categories
        action_features = [
            action_distribution[0],                      # No-op frequency
            np.sum(action_distribution[2:6]),           # Movement actions
            np.sum(action_distribution[10:14]),         # Shooting actions
            np.sum(action_distribution[1:2]),           # Fire action
            np.max(action_distribution),                # Most frequent action
            1.0 - np.max(action_distribution)           # Action diversity
        ]
        
        # Temporal patterns (4 features)
        episode_length = episode_data.get('episode_length', 1000)
        score = episode_data.get('score', 0)
        
        temporal_features = [
            episode_length / 18000,                     # Normalized survival
            score / max(episode_length, 1),             # Score efficiency  
            episode_data.get('early_score_ratio', 0.5), # Early vs late scoring
            episode_data.get('risk_events', 0) / max(episode_length / 1000, 1)  # Risk frequency
        ]
        
        # Spatial preferences (4 features)
        spatial_features = [
            np.mean(positions[:, 0]),                   # X preference
            np.mean(positions[:, 1]),                   # Y preference
            episode_data.get('edge_time_ratio', 0.5),   # Time near edges
            episode_data.get('center_time_ratio', 0.5)  # Time in center
        ]
        
        # Combine all features
        behavior = np.array(movement_features + action_features + temporal_features + spatial_features)
        
        # Normalize and handle NaN values
        behavior = np.nan_to_num(behavior, nan=0.5, posinf=1.0, neginf=0.0)
        behavior = np.clip(behavior, 0, 2)  # Reasonable bounds
        
        return behavior


class PlateauManager:
    """Advanced plateau detection and response system"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.fitness_history = deque(maxlen=config.plateau_detection_window)
        self.plateau_level = 0  # 0=no plateau, 1=mild, 2=moderate, 3=severe
        self.generations_since_improvement = 0
        self.best_fitness_ever = 0
        self.intervention_history = []
        
    def update(self, current_best_fitness: float, avg_fitness: float) -> Dict[str, Any]:
        """Update plateau status and recommend interventions"""
        
        self.fitness_history.append(current_best_fitness)
        
        # Check for meaningful improvement
        if current_best_fitness > self.best_fitness_ever + self.config.min_improvement_threshold:
            self.best_fitness_ever = current_best_fitness
            self.generations_since_improvement = 0
            self.plateau_level = 0
        else:
            self.generations_since_improvement += 1
        
        # Determine plateau severity
        plateau_detected = self.generations_since_improvement >= self.config.plateau_detection_window
        
        if plateau_detected:
            # Calculate plateau severity based on:
            # 1. How long we've been stuck
            # 2. Variance in recent fitness (low variance = stuck)
            # 3. Improvement rate trend
            
            duration_factor = min(self.generations_since_improvement / self.config.plateau_detection_window, 3.0)
            
            if len(self.fitness_history) >= 5:
                variance_factor = 1.0 / (1.0 + np.std(list(self.fitness_history)))
                trend_factor = self._calculate_trend_factor()
            else:
                variance_factor = 1.0
                trend_factor = 1.0
            
            severity_score = (duration_factor + variance_factor + trend_factor) / 3.0
            
            if severity_score < 1.0:
                self.plateau_level = 1  # Mild
            elif severity_score < 2.0:
                self.plateau_level = 2  # Moderate  
            else:
                self.plateau_level = 3  # Severe
        else:
            self.plateau_level = 0
        
        # Generate intervention recommendations
        interventions = self._generate_interventions()
        
        return {
            "plateau_detected": plateau_detected,
            "plateau_level": self.plateau_level,
            "generations_stuck": self.generations_since_improvement,
            "interventions": interventions,
            "severity_score": severity_score if plateau_detected else 0
        }
    
    def _calculate_trend_factor(self) -> float:
        """Calculate fitness trend (declining = worse plateau)"""
        if len(self.fitness_history) < 5:
            return 1.0
        
        recent_fitness = list(self.fitness_history)
        x = np.arange(len(recent_fitness))
        
        # Linear regression slope
        slope = np.polyfit(x, recent_fitness, 1)[0]
        
        # Negative slope = declining trend = worse plateau
        return max(0, -slope / 10.0) + 1.0
    
    def _generate_interventions(self) -> List[str]:
        """Generate appropriate interventions based on plateau level"""
        
        interventions = []
        
        if self.plateau_level == 1:  # Mild plateau
            interventions = [
                "increase_mutation_rate",
                "increase_exploration",
                "diversity_injection"
            ]
        
        elif self.plateau_level == 2:  # Moderate plateau
            interventions = [
                "increase_mutation_rate",
                "population_refresh",
                "increase_novelty_pressure",
                "curriculum_adjustment",
                "network_perturbation"
            ]
        
        elif self.plateau_level == 3:  # Severe plateau
            interventions = [
                "major_population_overhaul",
                "increase_population_size",
                "maximum_mutation",
                "reset_worst_performers",
                "architecture_modification",
                "speciation_reset"
            ]
        
        return interventions


class NoveltyArchive:
    """Manage novelty archive for diversity preservation"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.archive = []
        self.generation_added = []
        
    def add_behavior(self, behavior: np.ndarray, generation: int):
        """Add behavior to archive"""
        self.archive.append(behavior.copy())
        self.generation_added.append(generation)
        
        # Maintain archive size
        while len(self.archive) > self.config.novelty_archive_size:
            self.archive.pop(0)
            self.generation_added.pop(0)
    
    def calculate_novelty(self, behavior: np.ndarray) -> float:
        """Calculate novelty score for a behavior"""
        if len(self.archive) == 0:
            return 1.0
        
        # Calculate distances to all archived behaviors
        distances = []
        for archived_behavior in self.archive:
            distance = np.linalg.norm(behavior - archived_behavior)
            distances.append(distance)
        
        # Novelty is average distance to k nearest neighbors
        distances.sort()
        k = min(self.config.k_nearest_neighbors, len(distances))
        novelty = np.mean(distances[:k])
        
        return novelty
    
    def should_add_to_archive(self, behavior: np.ndarray, fitness: float) -> bool:
        """Decide whether to add behavior to archive"""
        novelty = self.calculate_novelty(behavior)
        
        # Add if highly novel OR high-performing
        return (novelty > self.config.behavioral_distance_threshold or 
                fitness > np.mean([b for b in self.archive]) if self.archive else True)


class SpeciesManager:
    """Manage species for diversity and specialized evolution"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.species = {}  # species_id -> {"members": [], "representative": network, "age": int}
        self.next_species_id = 0
        
    def assign_to_species(self, population: List) -> Dict[int, List]:
        """Assign population to species based on genetic similarity"""
        
        # Clear existing assignments
        for species_info in self.species.values():
            species_info["members"] = []
        
        # Assign each individual to a species
        for individual in population:
            assigned = False
            
            # Try to assign to existing species
            for species_id, species_info in self.species.items():
                if self._is_compatible(individual, species_info["representative"]):
                    species_info["members"].append(individual)
                    assigned = True
                    break
            
            # Create new species if no compatible species found
            if not assigned:
                self.species[self.next_species_id] = {
                    "members": [individual],
                    "representative": individual,
                    "age": 0
                }
                self.next_species_id += 1
        
        # Remove empty species and age existing species
        empty_species = [sid for sid, info in self.species.items() if len(info["members"]) == 0]
        for sid in empty_species:
            del self.species[sid]
        
        for species_info in self.species.values():
            species_info["age"] += 1
        
        return {sid: info["members"] for sid, info in self.species.items()}
    
    def _is_compatible(self, individual1, individual2) -> bool:
        """Check if two individuals belong to same species"""
        genetic_distance = self._calculate_genetic_distance(individual1, individual2)
        return genetic_distance < self.config.compatibility_threshold
    
    def _calculate_genetic_distance(self, ind1, ind2) -> float:
        """Calculate genetic distance between two networks"""
        total_distance = 0.0
        param_count = 0
        
        with torch.no_grad():
            for p1, p2 in zip(ind1.parameters(), ind2.parameters()):
                if p1.shape == p2.shape:
                    distance = torch.norm(p1 - p2).item()
                    total_distance += distance
                    param_count += p1.numel()
        
        return total_distance / max(param_count, 1)
    
    def get_species_statistics(self) -> Dict[str, Any]:
        """Get statistics about current species"""
        stats = {
            "num_species": len(self.species),
            "species_sizes": [len(info["members"]) for info in self.species.values()],
            "species_ages": [info["age"] for info in self.species.values()],
            "avg_species_size": np.mean([len(info["members"]) for info in self.species.values()]) if self.species else 0
        }
        return stats


class AdvancedEvolutionTrainer:
    """Enhanced trainer with all plateau-breaking strategies"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        
        # Initialize all advanced systems
        self.plateau_manager = PlateauManager(config)
        self.novelty_archive = NoveltyArchive(config)
        self.species_manager = SpeciesManager(config)
        
        # Evolution state
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.species_history = []
        
        # Current adaptive parameters
        self.current_mutation_rate = config.base_mutation_rate
        self.current_mutation_strength = config.base_mutation_strength
        self.current_dropout_rate = config.base_dropout_rate
        self.current_curriculum_stage = 0
        
        # Experience memory for behavioral analysis
        self.experience_memory = deque(maxlen=config.memory_size)
        
        logger.info(f"🚀 Advanced Plateau-Breaking Trainer Initialized")
        logger.info(f"   Population: {config.population_size}")
        logger.info(f"   Novelty search: {config.enable_novelty_search}")
        logger.info(f"   Speciation: {config.enable_speciation}")
        logger.info(f"   Curriculum: {config.enable_curriculum}")
        logger.info(f"   Multi-objective: {config.enable_multi_objective}")
    
    def evaluate_population_advanced(self) -> Tuple[float, float]:
        """Advanced evaluation with all plateau-breaking features"""
        
        logger.info(f"🧬 Advanced Evaluation - Generation {self.generation}")
        
        # Standard fitness evaluation
        evaluation_results = self._evaluate_population_standard()
        
        # Extract behavioral descriptors
        behavioral_data = []
        for i, result in enumerate(evaluation_results):
            behavior = BehavioralDescriptor.extract_behavior(result)
            behavioral_data.append(behavior)
            
            # Update experience memory
            experience = {
                'generation': self.generation,
                'individual_id': self.population[i].genome_id,
                'fitness': result['fitness'],
                'behavior': behavior,
                'episode_data': result
            }
            self.experience_memory.append(experience)
        
        # Calculate novelty scores
        novelty_scores = []
        for behavior in behavioral_data:
            novelty = self.novelty_archive.calculate_novelty(behavior)
            novelty_scores.append(novelty)
            
            # Add interesting behaviors to archive
            if self.novelty_archive.should_add_to_archive(behavior, 0):  # fitness added later
                self.novelty_archive.add_behavior(behavior, self.generation)
        
        # Apply multi-objective fitness if enabled
        if self.config.enable_multi_objective:
            self._apply_multi_objective_fitness(evaluation_results, novelty_scores)
        
        # Update individual fitness
        for i, (individual, result) in enumerate(zip(self.population, evaluation_results)):
            individual.fitness = result['fitness']
            individual.episode_rewards = result.get('episode_rewards', [])
        
        # Apply speciation if enabled
        if self.config.enable_speciation:
            species_assignments = self.species_manager.assign_to_species(self.population)
            self._apply_fitness_sharing(species_assignments)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        diversity_score = np.mean(novelty_scores) if novelty_scores else 0
        
        # Update histories
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity_score)
        
        if self.config.enable_speciation:
            species_stats = self.species_manager.get_species_statistics()
            self.species_history.append(species_stats)
        
        # Plateau detection and intervention
        plateau_info = self.plateau_manager.update(best_fitness, avg_fitness)
        
        if plateau_info["plateau_detected"]:
            self._apply_plateau_interventions(plateau_info)
        
        # Update curriculum if enabled
        if self.config.enable_curriculum:
            self._update_curriculum(best_fitness)
        
        # Log progress
        self._log_advanced_progress(best_fitness, avg_fitness, diversity_score, plateau_info)
        
        return best_fitness, avg_fitness
    
    def _evaluate_population_standard(self) -> List[Dict]:
        """Standard population evaluation (implement your existing logic)"""
        # This should integrate with your existing evaluate_individual function
        # For now, returning placeholder structure
        
        results = []
        for individual in self.population:
            # Simulate evaluation result structure
            result = {
                'genome_id': individual.genome_id,
                'fitness': random.uniform(100, 1000),  # Replace with actual evaluation
                'avg_reward': random.uniform(50, 500),
                'avg_length': random.uniform(1000, 8000),
                'episode_rewards': [random.uniform(0, 200) for _ in range(self.config.episodes_per_eval)],
                'actions': [random.randint(0, 13) for _ in range(100)],  # Sample actions
                'positions': [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(50)],
                'episode_length': random.uniform(1000, 8000),
                'score': random.uniform(50, 500),
                'early_score_ratio': random.uniform(0.2, 0.8),
                'risk_events': random.randint(0, 10),
                'edge_time_ratio': random.uniform(0.1, 0.4),
                'center_time_ratio': random.uniform(0.3, 0.7)
            }
            results.append(result)
        
        return results
    
    def _apply_multi_objective_fitness(self, results: List[Dict], novelty_scores: List[float]):
        """Apply multi-objective fitness combination"""
        
        for i, result in enumerate(results):
            # Original fitness components
            score = result.get('avg_reward', 0)
            survival = result.get('avg_length', 0)
            consistency = 1.0 / (1.0 + np.std(result.get('episode_rewards', [0])))
            novelty = novelty_scores[i] if i < len(novelty_scores) else 0
            
            # Calculate risk-taking bonus (calculated risks vs reckless behavior)
            risk_bonus = min(result.get('risk_events', 0) / 5.0, 2.0) * (survival / 10000)
            
            # Multi-objective combination
            combined_fitness = (
                self.config.score_weight * score +
                self.config.survival_weight * survival * 0.1 +
                self.config.consistency_weight * consistency * 100 +
                self.config.novelty_weight * novelty * 200 +
                self.config.risk_taking_weight * risk_bonus * 50
            )
            
            result['fitness'] = combined_fitness
    
    def _apply_fitness_sharing(self, species_assignments: Dict[int, List]):
        """Apply fitness sharing within species"""
        
        for species_id, members in species_assignments.items():
            if len(members) <= 1:
                continue
            
            # Reduce fitness based on species size (encourage diversity)
            sharing_factor = 1.0 / len(members)
            
            for individual in members:
                individual.fitness *= sharing_factor
    
    def _apply_plateau_interventions(self, plateau_info: Dict):
        """Apply interventions based on plateau severity"""
        
        interventions = plateau_info["interventions"]
        plateau_level = plateau_info["plateau_level"]
        
        logger.info(f"🚨 Plateau Level {plateau_level} - Applying interventions: {interventions}")
        
        if "increase_mutation_rate" in interventions:
            multiplier = 1.0 + (plateau_level * 0.3)
            self.current_mutation_rate = min(
                self.config.base_mutation_rate * multiplier,
                self.config.max_mutation_rate
            )
            self.current_mutation_strength = min(
                self.config.base_mutation_strength * multiplier,
                self.config.max_mutation_strength
            )
        
        if "population_refresh" in interventions:
            # Replace worst performers with new individuals
            refresh_count = int(len(self.population) * self.config.population_refresh_rate)
            self._refresh_population(refresh_count)
        
        if "increase_population_size" in interventions and self.config.enable_population_growth:
            # Grow population for more diversity
            current_size = len(self.population)
            new_size = min(
                int(current_size * self.config.population_growth_factor),
                self.config.max_population_size
            )
            self._grow_population(new_size - current_size)
        
        if "major_population_overhaul" in interventions:
            # Major intervention for severe plateaus
            self._major_population_overhaul()
        
        if "network_perturbation" in interventions:
            # Apply heavy mutation to bottom performers
            bottom_third = len(self.population) // 3
            for individual in self.population[-bottom_third:]:
                self._heavy_mutate_individual(individual)
        
        if "architecture_modification" in interventions:
            # Modify network architecture (increase dropout, etc.)
            self.current_dropout_rate = min(
                self.config.plateau_dropout_rate,
                self.current_dropout_rate * 1.5
            )
    
    def _refresh_population(self, count: int):
        """Replace worst performers with new random individuals"""
        if count <= 0:
            return
        
        # Remove worst performers
        self.population = self.population[:-count]
        
        # Add new random individuals
        for _ in range(count):
            # Create new individual (implement based on your network creation)
            new_individual = self._create_random_individual()
            self.population.append(new_individual)
        
        logger.info(f"🔄 Refreshed {count} individuals")
    
    def _grow_population(self, growth_count: int):
        """Grow population size during plateaus"""
        if growth_count <= 0:
            return
        
        for _ in range(growth_count):
            new_individual = self._create_random_individual()
            self.population.append(new_individual)
        
        logger.info(f"📈 Grew population by {growth_count} to {len(self.population)}")
    
    def _major_population_overhaul(self):
        """Major intervention for severe plateaus"""
        # Keep top 30%, replace the rest
        keep_count = len(self.population) // 3
        elite = self.population[:keep_count]
        
        # Create new population
        new_population = [ind.clone() for ind in elite]  # Keep elite
        
        # Add mutated versions of elite
        for _ in range(keep_count):
            mutated = random.choice(elite).clone()
            self._heavy_mutate_individual(mutated)
            new_population.append(mutated)
        
        # Fill rest with random individuals
        while len(new_population) < len(self.population):
            new_individual = self._create_random_individual()
            new_population.append(new_individual)
        
        self.population = new_population
        logger.info(f"💥 Major population overhaul completed")
    
    def _heavy_mutate_individual(self, individual):
        """Apply heavy mutation to individual"""
        with torch.no_grad():
            for param in individual.parameters():
                if random.random() < 0.7:  # High mutation probability
                    noise = torch.randn_like(param) * self.config.max_mutation_strength
                    param.add_(noise)
    
    def _create_random_individual(self):
        """Create new random individual (implement based on your network class)"""
        # This should create a new EnhancedAsteroidsNetwork
        # Placeholder implementation
        from your_network_module import EnhancedAsteroidsNetwork  # Replace with actual import
        return EnhancedAsteroidsNetwork(self.config)
    
    def _update_curriculum(self, best_fitness: float):
        """Update curriculum learning stage"""
        if self.current_curriculum_stage >= len(self.config.curriculum_stages) - 1:
            return
        
        current_stage = self.config.curriculum_stages[self.current_curriculum_stage]
        
        # Check if should advance to next stage
        target_reached = best_fitness >= current_stage["target_fitness"]
        time_limit = self.generation >= current_stage.get("generations", 999999)
        
        if target_reached or time_limit:
            self.current_curriculum_stage += 1
            next_stage = self.config.curriculum_stages[self.current_curriculum_stage]
            logger.info(f"🎓 Advanced to curriculum stage: {next_stage['name']}")
    
    def _log_advanced_progress(self, best_fitness: float, avg_fitness: float, 
                             diversity_score: float, plateau_info: Dict):
        """Enhanced logging with all metrics"""
        
        logger.info(f"Gen {self.generation}: Best={best_fitness:.1f}, Avg={avg_fitness:.1f}, "
                   f"Diversity={diversity_score:.3f}")
        
        if plateau_info["plateau_detected"]:
            logger.info(f"   🚨 Plateau Level {plateau_info['plateau_level']} "
                       f"({plateau_info['generations_stuck']} gens stuck)")
        
        if self.config.enable_speciation and self.species_history:
            species_stats = self.species_history[-1]
            logger.info(f"   🧬 Species: {species_stats['num_species']} active, "
                       f"avg size: {species_stats['avg_species_size']:.1f}")
        
        logger.info(f"   🎯 Mutation: rate={self.current_mutation_rate:.3f}, "
                   f"strength={self.current_mutation_strength:.3f}")
        
        if self.config.enable_curriculum:
            current_stage = self.config.curriculum_stages[self.current_curriculum_stage]
            logger.info(f"   📚 Curriculum: {current_stage['name']} stage")
    
    def create_next_generation_advanced(self):
        """Advanced generation creation with all plateau-breaking features"""
        
        # Get current curriculum stage for evaluation parameters
        if self.config.enable_curriculum:
            current_stage = self.config.curriculum_stages[self.current_curriculum_stage]
            # Could adjust selection pressure based on curriculum stage
        
        next_population = []
        
        # Species-based reproduction if enabled
        if self.config.enable_speciation:
            species_assignments = self.species_manager.assign_to_species(self.population)
            next_population = self._reproduce_by_species(species_assignments)
        else:
            next_population = self._reproduce_standard()
        
        # Apply age-layered evolution if enabled
        if self.config.enable_age_layered_evolution:
            next_population = self._apply_age_layered_selection(next_population)
        
        # Update population
        self.population = next_population
        self.generation += 1
        
        # Decay adaptive parameters after generation
        self._decay_adaptive_parameters()
    
    def _reproduce_by_species(self, species_assignments: Dict[int, List]) -> List:
        """Reproduction within species for diversity preservation"""
        
        next_population = []
        total_fitness = sum(ind.fitness for ind in self.population)
        
        for species_id, members in species_assignments.items():
            if len(members) == 0:
                continue
            
            # Calculate species fitness share
            species_fitness = sum(ind.fitness for ind in members)
            species_share = species_fitness / max(total_fitness, 1)
            
            # Determine offspring count for this species
            offspring_count = max(1, int(species_share * self.config.population_size))
            offspring_count = min(offspring_count, len(members) * 3)  # Cap growth
            
            # Elite preservation within species
            species_elite_count = min(self.config.species_elitism, len(members))
            species_elite = sorted(members, key=lambda x: x.fitness, reverse=True)[:species_elite_count]
            
            for elite in species_elite:
                next_population.append(elite.clone())
            
            # Generate remaining offspring for this species
            remaining = offspring_count - species_elite_count
            
            for _ in range(remaining):
                if len(members) >= 2 and random.random() > self.config.interspecies_mating_rate:
                    # Intra-species mating
                    parent1 = self._tournament_selection(members)
                    parent2 = self._tournament_selection(members)
                else:
                    # Inter-species mating (rare)
                    parent1 = self._tournament_selection(members)
                    parent2 = self._tournament_selection(self.population)
                
                offspring = self._create_offspring(parent1, parent2)
                next_population.append(offspring)
        
        # Ensure we have enough individuals
        while len(next_population) < self.config.population_size:
            parent1 = self._tournament_selection(self.population)
            parent2 = self._tournament_selection(self.population)
            offspring = self._create_offspring(parent1, parent2)
            next_population.append(offspring)
        
        # Trim if too many
        next_population = next_population[:self.config.population_size]
        
        return next_population
    
    def _reproduce_standard(self) -> List:
        """Standard reproduction without speciation"""
        
        next_population = []
        
        # Elite preservation
        elite_count = self.config.elite_size
        elites = self.population[:elite_count]
        
        for elite in elites:
            next_population.append(elite.clone())
        
        # Generate remaining offspring
        while len(next_population) < self.config.population_size:
            parent1 = self._tournament_selection(self.population)
            parent2 = self._tournament_selection(self.population)
            offspring = self._create_offspring(parent1, parent2)
            next_population.append(offspring)
        
        return next_population
    
    def _create_offspring(self, parent1, parent2):
        """Create offspring with advanced mutation strategies"""
        
        # Crossover
        if random.random() < 0.7:  # Crossover rate
            offspring = parent1.crossover(parent2)
        else:
            offspring = parent1.clone()
        
        # Apply adaptive mutation
        self._mutate_individual_adaptive(offspring)
        
        offspring.generation = self.generation + 1
        return offspring
    
    def _mutate_individual_adaptive(self, individual):
        """Apply adaptive mutation based on current state"""
        
        mutation_rate = self.current_mutation_rate
        mutation_strength = self.current_mutation_strength
        
        with torch.no_grad():
            for param in individual.parameters():
                if random.random() < mutation_rate:
                    # Adaptive noise based on parameter statistics
                    param_std = torch.std(param).item()
                    adaptive_strength = mutation_strength * (1.0 + param_std * 0.5)
                    
                    noise = torch.randn_like(param) * adaptive_strength
                    param.add_(noise)
        
        # Reset fitness
        individual.fitness = 0.0
        individual.episode_rewards = []
    
    def _tournament_selection(self, candidates: List, tournament_size: int = None):
        """Tournament selection with optional fitness sharing"""
        
        if tournament_size is None:
            tournament_size = self.config.tournament_size
        
        tournament_size = min(tournament_size, len(candidates))
        tournament = random.sample(candidates, tournament_size)
        
        # Standard tournament selection
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    def _apply_age_layered_selection(self, population: List) -> List:
        """Apply age-layered evolution principles"""
        
        if not self.config.enable_age_layered_evolution:
            return population
        
        # Group by age (generation)
        age_groups = defaultdict(list)
        for individual in population:
            age = getattr(individual, 'generation', 0)
            age_groups[age].append(individual)
        
        # Apply different selection pressures by age
        filtered_population = []
        
        for age, individuals in age_groups.items():
            if age == 0:  # Young individuals - give them a chance
                survival_rate = 0.8
            elif age < 5:  # Mature individuals - normal selection
                survival_rate = 0.6
            else:  # Old individuals - higher selection pressure
                survival_rate = 0.4
            
            survival_count = max(1, int(len(individuals) * survival_rate))
            survivors = sorted(individuals, key=lambda x: x.fitness, reverse=True)[:survival_count]
            filtered_population.extend(survivors)
        
        return filtered_population
    
    def _decay_adaptive_parameters(self):
        """Decay adaptive parameters after each generation"""
        
        # Decay mutation rates toward baseline
        self.current_mutation_rate = max(
            self.current_mutation_rate * self.config.mutation_decay_rate,
            self.config.base_mutation_rate
        )
        
        self.current_mutation_strength = max(
            self.current_mutation_strength * self.config.mutation_decay_rate,
            self.config.base_mutation_strength
        )
        
        # Decay dropout rate toward baseline
        self.current_dropout_rate = max(
            self.current_dropout_rate * 0.98,
            self.config.base_dropout_rate
        )
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        
        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(ind.fitness for ind in self.population) if self.population else 0,
            "avg_fitness": np.mean([ind.fitness for ind in self.population]) if self.population else 0,
            "fitness_std": np.std([ind.fitness for ind in self.population]) if self.population else 0,
            
            # Adaptive parameters
            "current_mutation_rate": self.current_mutation_rate,
            "current_mutation_strength": self.current_mutation_strength,
            "current_dropout_rate": self.current_dropout_rate,
            
            # Diversity metrics
            "diversity_score": self.diversity_history[-1] if self.diversity_history else 0,
            "novelty_archive_size": len(self.novelty_archive.archive),
            
            # Plateau information
            "generations_since_improvement": self.plateau_manager.generations_since_improvement,
            "plateau_level": self.plateau_manager.plateau_level,
            
            # Curriculum information
            "curriculum_stage": self.current_curriculum_stage,
            "curriculum_stage_name": self.config.curriculum_stages[self.current_curriculum_stage]["name"]
        }
        
        # Species information
        if self.config.enable_speciation and self.species_history:
            species_stats = self.species_history[-1]
            stats.update({
                "num_species": species_stats["num_species"],
                "avg_species_size": species_stats["avg_species_size"],
                "largest_species_size": max(species_stats["species_sizes"]) if species_stats["species_sizes"] else 0
            })
        
        return stats
    
    def save_advanced_checkpoint(self, generation: int, save_dir: Path):
        """Save comprehensive checkpoint with all advanced state"""
        
        # Prepare checkpoint data
        checkpoint = {
            # Basic evolution state
            "generation": generation,
            "population_size": len(self.population),
            "best_individual": self.population[0].to_cpu().state_dict() if self.population else None,
            "config": self.config.__dict__,
            
            # Fitness histories
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "diversity_history": self.diversity_history,
            "species_history": self.species_history,
            
            # Advanced system states
            "plateau_manager_state": {
                "fitness_history": list(self.plateau_manager.fitness_history),
                "plateau_level": self.plateau_manager.plateau_level,
                "generations_since_improvement": self.plateau_manager.generations_since_improvement,
                "best_fitness_ever": self.plateau_manager.best_fitness_ever
            },
            
            "novelty_archive_state": {
                "archive": [behavior.tolist() for behavior in self.novelty_archive.archive],
                "generation_added": self.novelty_archive.generation_added
            },
            
            "adaptive_parameters": {
                "current_mutation_rate": self.current_mutation_rate,
                "current_mutation_strength": self.current_mutation_strength,
                "current_dropout_rate": self.current_dropout_rate,
                "current_curriculum_stage": self.current_curriculum_stage
            },
            
            # Experience memory (sample to avoid huge files)
            "experience_memory": list(self.experience_memory)[-100:] if self.experience_memory else []
        }
        
        # Save checkpoint
        checkpoint_path = save_dir / f"advanced_checkpoint_gen_{generation:03d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save advanced statistics
        stats = self.get_advanced_statistics()
        stats_path = save_dir / f"advanced_stats_gen_{generation:03d}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Advanced checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_advanced_checkpoint(self, checkpoint_path: Path):
        """Load comprehensive checkpoint with all advanced state"""
        
        logger.info(f"📂 Loading advanced checkpoint: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore basic state
        self.generation = checkpoint["generation"]
        self.best_fitness_history = checkpoint["best_fitness_history"]
        self.avg_fitness_history = checkpoint["avg_fitness_history"]
        self.diversity_history = checkpoint.get("diversity_history", [])
        self.species_history = checkpoint.get("species_history", [])
        
        # Restore plateau manager state
        plateau_state = checkpoint.get("plateau_manager_state", {})
        self.plateau_manager.fitness_history = deque(
            plateau_state.get("fitness_history", []),
            maxlen=self.config.plateau_detection_window
        )
        self.plateau_manager.plateau_level = plateau_state.get("plateau_level", 0)
        self.plateau_manager.generations_since_improvement = plateau_state.get("generations_since_improvement", 0)
        self.plateau_manager.best_fitness_ever = plateau_state.get("best_fitness_ever", 0)
        
        # Restore novelty archive
        archive_state = checkpoint.get("novelty_archive_state", {})
        if archive_state.get("archive"):
            self.novelty_archive.archive = [np.array(behavior) for behavior in archive_state["archive"]]
            self.novelty_archive.generation_added = archive_state.get("generation_added", [])
        
        # Restore adaptive parameters
        adaptive_params = checkpoint.get("adaptive_parameters", {})
        self.current_mutation_rate = adaptive_params.get("current_mutation_rate", self.config.base_mutation_rate)
        self.current_mutation_strength = adaptive_params.get("current_mutation_strength", self.config.base_mutation_strength)
        self.current_dropout_rate = adaptive_params.get("current_dropout_rate", self.config.base_dropout_rate)
        self.current_curriculum_stage = adaptive_params.get("current_curriculum_stage", 0)
        
        # Restore experience memory
        experience_data = checkpoint.get("experience_memory", [])
        self.experience_memory = deque(experience_data, maxlen=self.config.memory_size)
        
        logger.info(f"✅ Advanced checkpoint loaded:")
        logger.info(f"   Generation: {self.generation}")
        logger.info(f"   Best fitness: {self.best_fitness_history[-1] if self.best_fitness_history else 0:.1f}")
        logger.info(f"   Plateau level: {self.plateau_manager.plateau_level}")
        logger.info(f"   Archive size: {len(self.novelty_archive.archive)}")
        logger.info(f"   Curriculum stage: {self.current_curriculum_stage}")


# INTEGRATION FUNCTIONS FOR YOUR EXISTING CODE

def integrate_with_existing_trainer(existing_trainer_class):
    """Function to integrate advanced features with your existing trainer"""
    
    class EnhancedTrainer(existing_trainer_class):
        """Enhanced version of your existing trainer with plateau-breaking"""
        
        def __init__(self, config):
            # Convert your config to advanced config
            advanced_config = AdvancedConfig(
                # Copy all your existing config values
                env_id=getattr(config, 'env_id', "ALE/Asteroids-v5"),
                population_size=getattr(config, 'population_size', 80),
                generations=getattr(config, 'generations', 2000),
                episodes_per_eval=getattr(config, 'episodes_per_eval', 3),
                max_steps_per_episode=getattr(config, 'max_steps_per_episode', 18000),
                
                # Enable advanced features
                enable_novelty_search=True,
                enable_speciation=True,
                enable_curriculum=True,
                enable_multi_objective=True,
                enable_population_growth=True
            )
            
            # Initialize both parent and advanced systems
            super().__init__(config)
            self.advanced_trainer = AdvancedEvolutionTrainer(advanced_config)
        
        def evaluate_population(self):
            """Enhanced evaluation with plateau-breaking"""
            
            # Use your existing evaluation but add advanced processing
            best_fitness, avg_fitness = super().evaluate_population()
            
            # Apply advanced plateau-breaking
            plateau_info = self.advanced_trainer.plateau_manager.update(best_fitness, avg_fitness)
            
            if plateau_info["plateau_detected"]:
                logger.info(f"🚨 Plateau detected - applying interventions")
                self._apply_simple_interventions(plateau_info)
            
            return best_fitness, avg_fitness
        
        def _apply_simple_interventions(self, plateau_info):
            """Apply simple plateau-breaking interventions"""
            
            interventions = plateau_info["interventions"]
            
            if "increase_mutation_rate" in interventions:
                # Increase your mutation rate
                self.config.mutation_rate = min(0.4, self.config.mutation_rate * 1.5)
                logger.info(f"📈 Increased mutation rate to {self.config.mutation_rate:.3f}")
            
            if "population_refresh" in interventions:
                # Replace worst 20% with new random individuals
                refresh_count = len(self.population) // 5
                self.population = self.population[:-refresh_count]
                
                for _ in range(refresh_count):
                    new_individual = self._create_new_individual()  # Use your existing method
                    self.population.append(new_individual)
                
                logger.info(f"🔄 Refreshed {refresh_count} individuals")
    
    return EnhancedTrainer


def quick_plateau_breaker_addon(trainer, population):
    """Quick addon function to apply plateau-breaking to existing trainer"""
    
    # Simple plateau detection
    if not hasattr(trainer, 'plateau_counter'):
        trainer.plateau_counter = 0
        trainer.last_best_fitness = 0
    
    current_best = max(ind.fitness for ind in population)
    
    if current_best > trainer.last_best_fitness + 20:  # Improvement threshold
        trainer.last_best_fitness = current_best
        trainer.plateau_counter = 0
    else:
        trainer.plateau_counter += 1
    
    # Apply interventions if plateau detected
    if trainer.plateau_counter >= 15:  # 15 generations without improvement
        logger.info(f"🚨 Plateau detected! Applying quick interventions...")
        
        # Intervention 1: Increase mutation
        trainer.config.mutation_rate = min(0.35, trainer.config.mutation_rate * 1.3)
        
        # Intervention 2: Population refresh
        refresh_count = len(population) // 4  # Replace worst 25%
        population = population[:-refresh_count]
        
        # Add new random individuals (you'll need to implement this)
        for _ in range(refresh_count):
            new_individual = trainer._create_random_individual()  # Your method
            population.append(new_individual)
        
        # Intervention 3: Heavy mutation on bottom performers
        bottom_quarter = len(population) // 4
        for individual in population[-bottom_quarter:]:
            trainer._heavy_mutate(individual)
        
        trainer.plateau_counter = 0  # Reset counter
        logger.info(f"✅ Applied interventions - mutation rate: {trainer.config.mutation_rate:.3f}")
    
    return population


# USAGE EXAMPLE
def apply_advanced_plateau_breaking():
    """Example of how to use the advanced plateau-breaking system"""
    
    # Create advanced configuration
    config = AdvancedConfig(
        population_size=80,
        generations=2000,
        episodes_per_eval=5,  # Increased for stability
        
        # Enable all advanced features
        enable_novelty_search=True,
        enable_speciation=True,
        enable_curriculum=True,
        enable_multi_objective=True,
        enable_population_growth=True,
        
        # Tune plateau detection
        plateau_detection_window=12,  # Detect plateaus faster
        min_improvement_threshold=20.0,  # Require meaningful improvement
        
        # Aggressive mutation during plateaus
        plateau_mutation_multiplier=2.0,
        max_mutation_rate=0.4,
        
        # Strong diversity preservation
        novelty_weight=0.25,
        diversity_preservation=True
    )
    
    # Create advanced trainer
    trainer = AdvancedEvolutionTrainer(config)
    
    # Training loop (replace your existing loop)
    for generation in range(config.generations):
        # Advanced evaluation with all plateau-breaking features
        best_fitness, avg_fitness = trainer.evaluate_population_advanced()
        
        # Create next generation with advanced reproduction
        trainer.create_next_generation_advanced()
        
        # Save advanced checkpoints
        if generation % 10 == 0:
            trainer.save_advanced_checkpoint(generation, Path("advanced_training"))
        
        # Early stopping with higher threshold
        if best_fitness > 1500:  # Higher target due to multi-objective fitness
            logger.info(f"🎯 Target achieved! Stopping at generation {generation}")
            break
    
    return trainer