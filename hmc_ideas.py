# IMPROVEMENT 1: Match PPO's advantage normalization exactly
def compute_advantages_and_returns(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages with PPO-style normalization"""
    advantages, returns = [], []
    advantage = 0
    
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 or dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantage = delta + gamma * gae_lambda * (1 - dones[t]) * advantage
        advantages.insert(0, advantage)
        returns.insert(0, advantage + values[t])
    
    advantages = np.array(advantages)
    returns = np.array(returns)
    
    # PPO-STYLE NORMALIZATION (exactly match PPO)
    if len(advantages) > 1 and np.std(advantages) > 1e-8:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    return advantages, returns


# IMPROVEMENT 2: Better performance tracking
class EnhancedExperimentTracker:
    """Enhanced tracking based on PPO blog insights"""
    
    def __init__(self, config):
        self.config = config
        self.save_dir = Path("bayesian_hmc_experiment")
        self.save_dir.mkdir(exist_ok=True)
        
        # Enhanced metrics
        self.results = {
            'ppo': {
                'episode_rewards': [], 'episode_lengths': [],
                'policy_losses': [], 'value_losses': [], 'clipped_fractions': [],
                'explained_variance': [], 'learning_rates': []
            },
            'hmc': {
                'episode_rewards': [], 'episode_lengths': [],
                'policy_losses': [], 'value_losses': [],
                'acceptance_rates': [], 'temperatures': [], 'step_sizes': [],
                'log_priors': [], 'log_likelihoods': [],
                'energy_changes': [], 'exploration_metrics': []
            }
        }
    
    def compute_performance_metrics(self, rewards, window=100):
        """Compute robust performance metrics"""
        if len(rewards) < window:
            return {}
        
        recent_rewards = rewards[-window:]
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'median_reward': np.median(recent_rewards),
            '25th_percentile': np.percentile(recent_rewards, 25),
            '75th_percentile': np.percentile(recent_rewards, 75),
            'stability': 1.0 / (1.0 + np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8))
        }
    
    def log_hmc_exploration_metrics(self, hmc_optimizer):
        """Track HMC-specific exploration metrics"""
        if len(hmc_optimizer.recent_acceptances) > 10:
            recent_acc = np.mean(hmc_optimizer.recent_acceptances[-20:])
            self.results['hmc']['exploration_metrics'].append({
                'recent_acceptance': recent_acc,
                'step_size': hmc_optimizer.step_size,
                'temperature': hmc_optimizer.temperature,
                'effective_sample_size': self._estimate_ess(hmc_optimizer)
            })
    
    def _estimate_ess(self, hmc_optimizer):
        """Estimate effective sample size for HMC"""
        if len(hmc_optimizer.acceptance_rates) < 50:
            return 0.0
        
        # Simple ESS estimate based on acceptance rate variance
        recent_rates = hmc_optimizer.acceptance_rates[-50:]
        autocorr = np.corrcoef(recent_rates[:-1], recent_rates[1:])[0,1]
        ess = len(recent_rates) / (1 + 2 * max(0, autocorr))
        return ess


# IMPROVEMENT 3: Hyperparameter sensitivity testing
def run_hyperparameter_sensitivity_test():
    """Test HMC robustness across hyperparameters"""
    
    # PPO baseline config
    ppo_configs = [
        {'learning_rate': 3e-4, 'clip_epsilon': 0.2},  # Standard
        {'learning_rate': 1e-4, 'clip_epsilon': 0.1},  # Conservative  
        {'learning_rate': 1e-3, 'clip_epsilon': 0.3},  # Aggressive
    ]
    
    # HMC test configs
    hmc_configs = [
        {'step_size': 0.005, 'target_acceptance': 0.5},  # Conservative
        {'step_size': 0.01, 'target_acceptance': 0.65},  # Standard
        {'step_size': 0.02, 'target_acceptance': 0.8},   # Aggressive
        {'step_size': 0.03, 'target_acceptance': 0.9},   # Very aggressive
    ]
    
    results = {'ppo': {}, 'hmc': {}}
    
    # Test each configuration
    for i, config in enumerate(hmc_configs):
        print(f"Testing HMC config {i+1}: {config}")
        # Run short experiment (200 episodes)
        final_performance = run_short_experiment('hmc', config)
        results['hmc'][f'config_{i+1}'] = {
            'config': config,
            'performance': final_performance,
            'stability': compute_stability_metric(final_performance)
        }
    
    return results


# IMPROVEMENT 4: Better likelihood computation matching PPO
def compute_log_likelihood_improved(self, states, actions, advantages, old_log_probs):
    """Improved likelihood matching PPO's computation exactly"""
    try:
        with torch.no_grad():
            logits, values = self.network(states)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # EXACT PPO-style advantage usage (no tanh, just direct normalization)
        # This should match your PPO implementation exactly
        policy_objective = (advantages.detach() * action_log_probs).mean()
        
        # PPO-style entropy calculation
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Combine exactly like PPO does
        log_likelihood = policy_objective + 0.01 * entropy
        
        # Only clip for extreme numerical issues
        log_likelihood = torch.clamp(log_likelihood, -100, 100)
        
        return log_likelihood
        
    except Exception as e:
        logger.warning(f"Likelihood computation failed: {e}")
        return torch.tensor(0.0, dtype=torch.float32)


# IMPROVEMENT 5: Environment consistency
def create_matched_environment(config, seed=None):
    """Create environment exactly matching PPO blog recommendations"""
    
    # Use exact same preprocessing as successful PPO implementations
    env = gym.make(config.env_id, render_mode=None)
    
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        
    # Standard Atari preprocessing (exactly match PPO)
    env = AtariPreprocessing(
        env,
        noop_max=30,           # PPO standard
        frame_skip=4,          # PPO standard
        screen_size=84,        # PPO standard
        terminal_on_life_loss=True,  # PPO uses this
        grayscale_obs=True,
        scale_obs=False
    )
    
    env = FrameStack(env, config.frame_stack)
    return env


# IMPROVEMENT 6: Proper statistical comparison
def statistical_comparison(ppo_rewards, hmc_rewards, window=100):
    """Proper statistical comparison of methods"""
    from scipy import stats
    
    if len(ppo_rewards) < window or len(hmc_rewards) < window:
        return None
    
    ppo_recent = ppo_rewards[-window:]
    hmc_recent = hmc_rewards[-window:]
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(hmc_recent, ppo_recent, equal_var=False)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = stats.mannwhitneyu(hmc_recent, ppo_recent, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(ppo_recent) + np.var(hmc_recent)) / 2)
    cohens_d = (np.mean(hmc_recent) - np.mean(ppo_recent)) / pooled_std
    
    return {
        'hmc_mean': np.mean(hmc_recent),
        'ppo_mean': np.mean(ppo_recent),
        'difference': np.mean(hmc_recent) - np.mean(ppo_recent),
        'cohens_d': cohens_d,
        't_test_p': p_value,
        'mannwhitney_p': u_p_value,
        'significantly_different': p_value < 0.05,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }