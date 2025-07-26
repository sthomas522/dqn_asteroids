#!/usr/bin/env python3
"""
Easy Plateau-Breaking Integration for Your Existing Asteroids Code
Drop this into your existing trainer with minimal changes required
"""

import numpy as np
import torch
import random
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class PlateauBreakerConfig:
    """Simple configuration for plateau-breaking features"""
    
    # Plateau Detection
    plateau_window: int = 15                    # Generations to detect plateau
    improvement_threshold: float = 20.0         # Minimum improvement required
    
    # Mutation Scaling
    base_mutation_rate: float = 0.15
    max_mutation_rate: float = 0.4
    mutation_multiplier: float = 1.5            # How much to increase during plateau
    
    # Population Management
    refresh_percentage: float = 0.25            # Replace 25% worst performers
    heavy_mutation_percentage: float = 0.3      # Apply heavy mutation to 30% worst
    
    # Diversity Injection
    enable_diversity_injection: bool = True
    diversity_injection_rate: float = 0.15      # Replace 15% with random individuals
    
    # Experience-based improvements
    enable_behavioral_diversity: bool = True
    action_diversity_weight: float = 0.1        # Bonus for diverse action patterns


class SimplePlateauBreaker:
    """Simple but effective plateau-breaking system"""
    
    def __init__(self, config: PlateauBreakerConfig):
        self.config = config
        self.fitness_history = deque(maxlen=config.plateau_window)
        self.best_fitness = 0
        self.generations_stuck = 0
        self.intervention_count = 0
        
        # Track behavioral diversity
        self.action_patterns = deque(maxlen=50)  # Store recent action patterns
        
        logger.info("🚀 Simple Plateau Breaker initialized")
        logger.info(f"   Plateau window: {config.plateau_window} generations")
        logger.info(f"   Improvement threshold: {config.improvement_threshold}")
    
    def check_plateau(self, current_best_fitness: float) -> Dict[str, Any]:
        """Check if we're in a plateau and return intervention recommendations"""
        
        self.fitness_history.append(current_best_fitness)
        
        # Check for meaningful improvement
        if current_best_fitness > self.best_fitness + self.config.improvement_threshold:
            self.best_fitness = current_best_fitness
            self.generations_stuck = 0
            plateau_detected = False
        else:
            self.generations_stuck += 1
            plateau_detected = self.generations_stuck >= self.config.plateau_window
        
        # Determine intervention level
        if plateau_detected:
            if self.generations_stuck < self.config.plateau_window * 1.5:
                intervention_level = "mild"
            elif self.generations_stuck < self.config.plateau_window * 2:
                intervention_level = "moderate"
            else:
                intervention_level = "severe"
        else:
            intervention_level = "none"
        
        return {
            "plateau_detected": plateau_detected,
            "intervention_level": intervention_level,
            "generations_stuck": self.generations_stuck,
            "best_fitness": self.best_fitness,
            "recent_variance": np.std(list(self.fitness_history)) if len(self.fitness_history) > 3 else 0
        }
    
    def apply_interventions(self, population: List, trainer, plateau_info: Dict) -> List:
        """Apply plateau-breaking interventions to population"""
        
        if not plateau_info["plateau_detected"]:
            return population
        
        level = plateau_info["intervention_level"]
        self.intervention_count += 1
        
        logger.info(f"🚨 Plateau intervention #{self.intervention_count} - Level: {level}")
        logger.info(f"   Stuck for {plateau_info['generations_stuck']} generations")
        
        if level == "mild":
            population = self._mild_intervention(population, trainer)
        elif level == "moderate":
            population = self._moderate_intervention(population, trainer)
        elif level == "severe":
            population = self._severe_intervention(population, trainer)
        
        return population
    
    def _mild_intervention(self, population: List, trainer) -> List:
        """Mild intervention for early plateau detection"""
        
        # 1. Increase mutation rate
        old_rate = trainer.config.mutation_rate
        trainer.config.mutation_rate = min(
            old_rate * self.config.mutation_multiplier,
            self.config.max_mutation_rate
        )
        
        # 2. Apply extra mutation to worst performers
        worst_count = int(len(population) * 0.2)  # Bottom 20%
        for individual in population[-worst_count:]:
            self._extra_mutate(individual, strength=0.15)
        
        logger.info(f"   🎯 Mild: Mutation {old_rate:.3f} → {trainer.config.mutation_rate:.3f}")
        logger.info(f"   🎯 Extra mutation applied to {worst_count} worst performers")
        
        return population
    
    def _moderate_intervention(self, population: List, trainer) -> List:
        """Moderate intervention for persistent plateaus"""
        
        # 1. Stronger mutation increase
        old_rate = trainer.config.mutation_rate
        trainer.config.mutation_rate = min(
            old_rate * (self.config.mutation_multiplier * 1.5),
            self.config.max_mutation_rate
        )
        
        # 2. Population refresh - replace worst performers
        refresh_count = int(len(population) * self.config.refresh_percentage)
        population = population[:-refresh_count]  # Remove worst
        
        # 3. Add new random individuals
        for _ in range(refresh_count):
            new_individual = self._create_random_individual(trainer)
            population.append(new_individual)
        
        # 4. Heavy mutation on bottom third
        heavy_mut_count = int(len(population) * self.config.heavy_mutation_percentage)
        for individual in population[-heavy_mut_count:]:
            self._heavy_mutate(individual)
        
        logger.info(f"   🔄 Moderate: Refreshed {refresh_count} individuals")
        logger.info(f"   🎯 Heavy mutation on {heavy_mut_count} individuals")
        logger.info(f"   📈 Mutation rate: {trainer.config.mutation_rate:.3f}")
        
        return population
    
    def _severe_intervention(self, population: List, trainer) -> List:
        """Severe intervention for stubborn plateaus"""
        
        # 1. Maximum mutation rate
        trainer.config.mutation_rate = self.config.max_mutation_rate
        
        # 2. Major population overhaul - keep only top 30%
        keep_count = int(len(population) * 0.3)
        elite = population[:keep_count]
        
        new_population = []
        
        # Keep elite
        for individual in elite:
            new_population.append(individual.clone())
        
        # Create mutated versions of elite
        for individual in elite:
            mutated = individual.clone()
            self._heavy_mutate(mutated)
            new_population.append(mutated)
        
        # Fill rest with completely new random individuals
        while len(new_population) < len(population):
            new_individual = self._create_random_individual(trainer)
            new_population.append(new_individual)
        
        logger.info(f"   💥 Severe: Major overhaul - kept {keep_count} elite")
        logger.info(f"   🎲 Created {len(population) - len(new_population)} new random individuals")
        logger.info(f"   ⚡ Maximum mutation rate: {trainer.config.mutation_rate:.3f}")
        
        return new_population
    
    def _extra_mutate(self, individual, strength: float = 0.1):
        """Apply extra mutation to individual"""
        with torch.no_grad():
            for param in individual.parameters():
                if random.random() < 0.6:  # 60% chance per parameter
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
    
    def _heavy_mutate(self, individual, strength: float = 0.2):
        """Apply heavy mutation to individual"""
        with torch.no_grad():
            for param in individual.parameters():
                if random.random() < 0.8:  # 80% chance per parameter
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
        
        # Reset fitness
        individual.fitness = 0.0
        individual.episode_rewards = []
    
    def _create_random_individual(self, trainer):
        """Create new random individual using trainer's network class"""
        # This should use your existing network creation method
        # Replace this with your actual EnhancedAsteroidsNetwork creation
        try:
            # Try to use your existing network creation
            from enhanced_asteroids_evolution import EnhancedAsteroidsNetwork
            new_individual = EnhancedAsteroidsNetwork(trainer.config)
            new_individual.generation = trainer.generation + 1
            return new_individual
        except ImportError:
            # Fallback - you'll need to customize this
            logger.warning("Could not import EnhancedAsteroidsNetwork - using placeholder")
            return None
    
    def add_behavioral_diversity_bonus(self, population: List, episode_data_list: List[Dict]) -> List:
        """Add behavioral diversity bonus to fitness"""
        
        if not self.config.enable_behavioral_diversity:
            return population
        
        # Calculate action diversity scores
        diversity_scores = []
        
        for i, episode_data in enumerate(episode_data_list):
            actions = episode_data.get('actions', [])
            if len(actions) == 0:
                diversity_scores.append(0)
                continue
            
            # Calculate action pattern diversity
            action_counts = np.bincount(actions, minlength=14)  # Asteroids has 14 actions
            action_probs = action_counts / len(actions)
            
            # Entropy-based diversity score
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            max_entropy = np.log(14)  # Maximum possible entropy
            diversity_score = entropy / max_entropy
            
            diversity_scores.append(diversity_score)
            
            # Store action pattern for future comparison
            pattern = {
                'generation': getattr(self, 'current_generation', 0),
                'action_distribution': action_probs,
                'diversity_score': diversity_score
            }
            self.action_patterns.append(pattern)
        
        # Apply diversity bonus to fitness
        for i, individual in enumerate(population):
            if i < len(diversity_scores):
                diversity_bonus = diversity_scores[i] * self.config.action_diversity_weight * 100
                individual.fitness += diversity_bonus
        
        return population


class EnhancedEvaluationWrapper:
    """Wrapper to enhance your existing evaluation with behavioral analysis"""
    
    def __init__(self, original_evaluate_func):
        self.original_evaluate = original_evaluate_func
        self.action_tracker = {}
        
    def enhanced_evaluate_individual(self, network_state, config, episodes, seed, genome_id):
        """Enhanced evaluation that tracks behavioral data"""
        
        # Run original evaluation
        result = self.original_evaluate(network_state, config, episodes, seed, genome_id)
        
        # Add behavioral tracking (simplified version)
        # In a real implementation, you'd modify your evaluation loop to track:
        # - Action sequences
        # - Position trajectories  
        # - Risk-taking behaviors
        # - Movement patterns
        
        # For now, add some simulated behavioral data
        episode_length = result.get('avg_length', 1000)
        score = result.get('avg_reward', 0)
        
        # Simulate action diversity (you'd track this during actual evaluation)
        action_diversity = min(1.0, score / 500.0)  # Higher score = more diverse actions
        
        # Add behavioral features to result
        result.update({
            'actions': [random.randint(0, 13) for _ in range(min(100, int(episode_length / 50)))],
            'positions': [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(20)],
            'action_diversity': action_diversity,
            'risk_events': random.randint(0, 5),
            'movement_efficiency': min(1.0, episode_length / 10000),
            'early_score_ratio': random.uniform(0.2, 0.8)
        })
        
        return result


# INTEGRATION FUNCTIONS FOR YOUR EXISTING CODE

def add_plateau_breaking_to_trainer(trainer_class):
    """Decorator to add plateau-breaking to your existing trainer class"""
    
    class PlateauEnhancedTrainer(trainer_class):
        """Your existing trainer enhanced with plateau-breaking"""
        
        def __init__(self, config):
            super().__init__(config)
            
            # Add plateau breaker
            plateau_config = PlateauBreakerConfig(
                plateau_window=15,
                improvement_threshold=25.0,  # Adjust based on your typical improvements
                base_mutation_rate=getattr(config, 'mutation_rate', 0.15),
                max_mutation_rate=0.4,
                refresh_percentage=0.25,
                enable_behavioral_diversity=True
            )
            
            self.plateau_breaker = SimplePlateauBreaker(plateau_config)
            
            # Track enhanced evaluation
            if hasattr(self, 'evaluate_individual'):
                self.enhanced_evaluator = EnhancedEvaluationWrapper(self.evaluate_individual)
            
            logger.info("✅ Plateau-breaking added to existing trainer")
        
        def evaluate_population(self):
            """Enhanced evaluation with plateau detection"""
            
            # Run your existing evaluation
            best_fitness, avg_fitness = super().evaluate_population()
            
            # Check for plateau and apply interventions
            plateau_info = self.plateau_breaker.check_plateau(best_fitness)
            
            if plateau_info["plateau_detected"]:
                self.population = self.plateau_breaker.apply_interventions(
                    self.population, self, plateau_info
                )
                
                # Re-sort population after interventions
                self.population.sort(key=lambda x: x.fitness, reverse=True)
                
                # Update fitness statistics after interventions
                new_fitnesses = [ind.fitness for ind in self.population]
                best_fitness = max(new_fitnesses)
                avg_fitness = np.mean(new_fitnesses)
            
            return best_fitness, avg_fitness
        
        def create_next_generation(self):
            """Enhanced generation creation with diversity preservation"""
            
            # Run your existing generation creation
            super().create_next_generation()
            
            # Apply behavioral diversity bonus if enabled
            if (self.plateau_breaker.config.enable_behavioral_diversity and 
                hasattr(self, 'last_episode_data')):
                self.population = self.plateau_breaker.add_behavioral_diversity_bonus(
                    self.population, self.last_episode_data
                )
    
    return PlateauEnhancedTrainer


def simple_plateau_intervention(trainer, population, generation):
    """Simple function to add plateau-breaking to existing training loop"""
    
    # Initialize plateau tracker if not exists
    if not hasattr(trainer, 'plateau_tracker'):
        trainer.plateau_tracker = {
            'best_fitness_history': [],
            'plateau_counter': 0,
            'best_ever': 0,
            'interventions_applied': 0
        }
    
    # Get current best fitness
    current_best = max(ind.fitness for ind in population) if population else 0
    trainer.plateau_tracker['best_fitness_history'].append(current_best)
    
    # Check for improvement
    if current_best > trainer.plateau_tracker['best_ever'] + 20:  # 20 point improvement threshold
        trainer.plateau_tracker['best_ever'] = current_best
        trainer.plateau_tracker['plateau_counter'] = 0
    else:
        trainer.plateau_tracker['plateau_counter'] += 1
    
    # Apply intervention if plateau detected
    if trainer.plateau_tracker['plateau_counter'] >= 15:  # 15 generations stuck
        
        trainer.plateau_tracker['interventions_applied'] += 1
        intervention_num = trainer.plateau_tracker['interventions_applied']
        
        logger.info(f"🚨 Plateau detected at generation {generation}!")
        logger.info(f"   Best fitness stuck at {current_best:.1f} for {trainer.plateau_tracker['plateau_counter']} generations")
        logger.info(f"   Applying intervention #{intervention_num}")
        
        # Intervention 1: Increase mutation rate
        old_mutation = trainer.config.mutation_rate
        trainer.config.mutation_rate = min(0.4, old_mutation * 1.5)
        
        # Intervention 2: Population refresh
        if intervention_num % 2 == 1:  # Every other intervention
            refresh_count = len(population) // 4  # Replace worst 25%
            
            # Keep best 75%
            population = population[:-refresh_count]
            
            # Add new random individuals
            for _ in range(refresh_count):
                try:
                    # Use your existing network creation method
                    new_individual = trainer._create_new_individual()  # You may need to adjust method name
                    population.append(new_individual)
                except AttributeError:
                    logger.warning("Could not create new individual - skipping population refresh")
                    break
            
            logger.info(f"   🔄 Refreshed {refresh_count} worst performers")
        
        # Intervention 3: Heavy mutation on worst performers
        else:
            heavy_mut_count = len(population) // 3  # Bottom third
            for individual in population[-heavy_mut_count:]:
                # Apply heavy mutation
                with torch.no_grad():
                    for param in individual.parameters():
                        if random.random() < 0.7:
                            noise = torch.randn_like(param) * 0.2
                            param.add_(noise)
                
                # Reset fitness
                individual.fitness = 0.0
                individual.episode_rewards = []
            
            logger.info(f"   ⚡ Heavy mutation on {heavy_mut_count} worst performers")
        
        logger.info(f"   📈 Mutation rate: {old_mutation:.3f} → {trainer.config.mutation_rate:.3f}")
        
        # Reset plateau counter
        trainer.plateau_tracker['plateau_counter'] = 0
    
    return population


# EASY INTEGRATION EXAMPLE

def integrate_plateau_breaking_easily():
    """Example showing how to easily integrate with your existing code"""
    
    # METHOD 1: Minimal integration - just add this to your training loop
    """
    # In your existing training loop, add this after evaluate_population():
    
    for generation in range(config.generations):
        best_fitness, avg_fitness = trainer.evaluate_population()
        
        # ADD THIS LINE:
        trainer.population = simple_plateau_intervention(trainer, trainer.population, generation)
        
        trainer.create_next_generation()
        # ... rest of your existing code
    """
    
    # METHOD 2: Enhance your trainer class
    """
    # Replace your trainer class instantiation:
    
    # OLD:
    # trainer = EvolutionTrainer(config)
    
    # NEW:
    EnhancedTrainer = add_plateau_breaking_to_trainer(EvolutionTrainer)
    trainer = EnhancedTrainer(config)
    """
    
    # METHOD 3: Manual integration with full control
    """
    # Create plateau breaker separately:
    
    plateau_config = PlateauBreakerConfig(
        plateau_window=12,  # Detect faster
        improvement_threshold=30.0,  # Require bigger improvements
        max_mutation_rate=0.45,  # Allow higher mutation
        refresh_percentage=0.3   # Replace more individuals
    )
    
    plateau_breaker = SimplePlateauBreaker(plateau_config)
    
    # In your training loop:
    for generation in range(config.generations):
        best_fitness, avg_fitness = trainer.evaluate_population()
        
        plateau_info = plateau_breaker.check_plateau(best_fitness)
        if plateau_info["plateau_detected"]:
            trainer.population = plateau_breaker.apply_interventions(
                trainer.population, trainer, plateau_info
            )
        
        trainer.create_next_generation()
    """
    
    pass


# CONFIGURATION TUNING GUIDE

def get_recommended_plateau_configs():
    """Get recommended configurations for different scenarios"""
    
    configs = {
        "conservative": PlateauBreakerConfig(
            plateau_window=20,           # Wait longer before intervening
            improvement_threshold=15.0,  # Lower threshold
            max_mutation_rate=0.3,       # Conservative mutation
            refresh_percentage=0.15,     # Small population refresh
        ),
        
        "aggressive": PlateauBreakerConfig(
            plateau_window=10,           # Intervene quickly
            improvement_threshold=30.0,  # Require significant improvement
            max_mutation_rate=0.5,       # High mutation allowed
            refresh_percentage=0.4,      # Large population refresh
        ),
        
        "balanced": PlateauBreakerConfig(
            plateau_window=15,           # Moderate detection time
            improvement_threshold=25.0,  # Reasonable improvement required
            max_mutation_rate=0.4,       # Balanced mutation
            refresh_percentage=0.25,     # Quarter population refresh
        ),
        
        "your_current_situation": PlateauBreakerConfig(
            plateau_window=12,           # Detect plateaus faster given your current state
            improvement_threshold=40.0,  # You need bigger improvements at 700-900 fitness
            max_mutation_rate=0.45,      # Allow aggressive mutation
            refresh_percentage=0.3,      # Significant population refresh
            heavy_mutation_percentage=0.4,  # Heavy mutation on more individuals
            enable_behavioral_diversity=True,  # Add behavioral diversity
        )
    }
    
    return configs


# QUICK START FUNCTION

def quick_start_plateau_breaking(existing_trainer, config_type="your_current_situation"):
    """Quick start function to add plateau-breaking to your existing setup"""
    
    logger.info("🚀 Quick Start: Adding Plateau-Breaking to Your Trainer")
    
    # Get recommended config
    configs = get_recommended_plateau_configs()
    plateau_config = configs[config_type]
    
    # Create plateau breaker
    plateau_breaker = SimplePlateauBreaker(plateau_config)
    
    # Add to trainer
    existing_trainer.plateau_breaker = plateau_breaker
    
    # Enhance the evaluate_population method
    original_evaluate = existing_trainer.evaluate_population
    
    def enhanced_evaluate_population():
        best_fitness, avg_fitness = original_evaluate()
        
        # Apply plateau breaking
        plateau_info = plateau_breaker.check_plateau(best_fitness)
        if plateau_info["plateau_detected"]:
            existing_trainer.population = plateau_breaker.apply_interventions(
                existing_trainer.population, existing_trainer, plateau_info
            )
            
            # Update statistics
            new_fitnesses = [ind.fitness for ind in existing_trainer.population]
            best_fitness = max(new_fitnesses)
            avg_fitness = np.mean(new_fitnesses)
        
        return best_fitness, avg_fitness
    
    # Replace method
    existing_trainer.evaluate_population = enhanced_evaluate_population
    
    logger.info(f"✅ Plateau-breaking added with '{config_type}' configuration")
    logger.info(f"   Plateau detection: {plateau_config.plateau_window} generations")
    logger.info(f"   Improvement threshold: {plateau_config.improvement_threshold}")
    logger.info(f"   Max mutation rate: {plateau_config.max_mutation_rate}")
    
    return existing_trainer


if __name__ == "__main__":
    print("🚀 Easy Plateau-Breaking Integration")
    print("=" * 50)
    print()
    print("QUICK INTEGRATION OPTIONS:")
    print()
    print("1. MINIMAL (add 1 line to training loop):")
    print("   trainer.population = simple_plateau_intervention(trainer, trainer.population, generation)")
    print()
    print("2. ENHANCED TRAINER (replace trainer class):")
    print("   EnhancedTrainer = add_plateau_breaking_to_trainer(YourTrainer)")
    print("   trainer = EnhancedTrainer(config)")
    print()
    print("3. QUICK START (enhance existing trainer):")
    print("   trainer = quick_start_plateau_breaking(trainer)")
    print()
    print("4. FULL CONTROL (manual integration):")
    print("   plateau_breaker = SimplePlateauBreaker(config)")
    print("   # Use in training loop")
    print()
    print("Choose the method that best fits your existing code!")