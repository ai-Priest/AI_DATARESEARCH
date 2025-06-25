"""
Advanced Hyperparameter Tuning for Neural Ranking
Comprehensive grid search and optimization for achieving 70%+ NDCG@3 performance.
"""

import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Advanced hyperparameter tuning with grid search and early stopping."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.results_history = []
        self.best_params = None
        self.best_score = 0.0
        
        # Define hyperparameter search space
        self.param_space = {
            'learning_rate': [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'hidden_dim': [512, 768, 1024],
            'attention_heads': [8, 12, 16],
            'weight_decay': [1e-4, 1e-3, 5e-3, 1e-2],
            'batch_size': [16, 32, 64],
            'warmup_steps': [100, 200, 500],
            'gradient_clip': [0.5, 1.0, 2.0]
        }
        
        # Focused search space for 70%+ target
        self.focused_space = {
            'learning_rate': [3e-4, 5e-4, 7e-4],
            'dropout_rate': [0.1, 0.2, 0.3],
            'hidden_dim': [768, 1024],
            'attention_heads': [12, 16],
            'weight_decay': [1e-3, 5e-3],
            'batch_size': [16, 32],
            'warmup_steps': [200, 500],
            'gradient_clip': [1.0, 2.0]
        }
    
    def grid_search(self, train_func, data_loaders: Dict, 
                   target_ndcg: float = 0.70, max_trials: int = 50,
                   use_focused_space: bool = True) -> Dict:
        """Perform grid search to find optimal hyperparameters."""
        
        logger.info(f"🔍 Starting hyperparameter grid search...")
        logger.info(f"🎯 Target NDCG@3: {target_ndcg}")
        logger.info(f"📊 Max trials: {max_trials}")
        
        # Choose search space
        search_space = self.focused_space if use_focused_space else self.param_space
        
        # Generate parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        # Limit to max_trials
        if len(all_combinations) > max_trials:
            # Use smart sampling for large spaces
            combinations = self._smart_sample_combinations(all_combinations, max_trials)
        else:
            combinations = all_combinations
        
        logger.info(f"🧪 Testing {len(combinations)} parameter combinations...")
        
        best_result = None
        target_achieved = False
        
        for i, param_combination in enumerate(combinations):
            logger.info(f"\n🔬 Trial {i+1}/{len(combinations)}")
            
            # Create parameter dictionary
            params = dict(zip(param_names, param_combination))
            logger.info(f"📋 Parameters: {params}")
            
            try:
                # Train model with these parameters
                start_time = time.time()
                result = train_func(params, data_loaders)
                training_time = time.time() - start_time
                
                # Extract performance metrics
                ndcg_3 = result.get('ndcg_at_3', 0.0)
                accuracy = result.get('accuracy', 0.0)
                f1_score = result.get('f1_score', 0.0)
                
                # Create trial result
                trial_result = {
                    'trial_id': i + 1,
                    'parameters': params,
                    'performance': {
                        'ndcg_at_3': ndcg_3,
                        'accuracy': accuracy,
                        'f1_score': f1_score
                    },
                    'training_time': training_time,
                    'timestamp': datetime.now().isoformat(),
                    'target_achieved': ndcg_3 >= target_ndcg
                }
                
                self.results_history.append(trial_result)
                
                # Update best result
                if ndcg_3 > self.best_score:
                    self.best_score = ndcg_3
                    self.best_params = params.copy()
                    best_result = trial_result.copy()
                    
                    logger.info(f"🏆 New best score: {ndcg_3:.4f}")
                    
                    # Check if target achieved
                    if ndcg_3 >= target_ndcg:
                        target_achieved = True
                        logger.info(f"🎯 Target NDCG@3 ≥ {target_ndcg} achieved!")
                        
                        # Continue for a few more trials to see if we can do better
                        if i >= 10:  # At least 10 trials before early stopping
                            logger.info("🏁 Early stopping - target achieved!")
                            break
                
                logger.info(f"📊 NDCG@3: {ndcg_3:.4f}, Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Trial {i+1} failed: {e}")
                continue
        
        # Compile final results
        final_results = {
            'best_parameters': self.best_params,
            'best_performance': {
                'ndcg_at_3': self.best_score,
                'target_achieved': target_achieved
            },
            'optimization_summary': {
                'total_trials': len(self.results_history),
                'target_ndcg': target_ndcg,
                'improvement_achieved': self.best_score - 0.60,  # From baseline
                'search_space_used': 'focused' if use_focused_space else 'full'
            },
            'all_trials': self.results_history
        }
        
        # Save results
        self._save_tuning_results(final_results)
        
        logger.info(f"\n🎉 Hyperparameter tuning complete!")
        logger.info(f"🏆 Best NDCG@3: {self.best_score:.4f}")
        logger.info(f"🎯 Target achieved: {'✅' if target_achieved else '❌'}")
        
        return final_results
    
    def _smart_sample_combinations(self, all_combinations: List, max_trials: int) -> List:
        """Smart sampling of parameter combinations for large search spaces."""
        
        # Use stratified sampling with some randomness
        step = len(all_combinations) // max_trials
        
        sampled = []
        for i in range(0, len(all_combinations), step):
            if len(sampled) >= max_trials:
                break
            sampled.append(all_combinations[i])
        
        # Add some random samples
        remaining_slots = max_trials - len(sampled)
        if remaining_slots > 0:
            import random
            remaining_combinations = [c for c in all_combinations if c not in sampled]
            random_samples = random.sample(
                remaining_combinations, 
                min(remaining_slots, len(remaining_combinations))
            )
            sampled.extend(random_samples)
        
        return sampled[:max_trials]
    
    def bayesian_optimization(self, train_func, data_loaders: Dict,
                            target_ndcg: float = 0.70, max_iterations: int = 30) -> Dict:
        """Bayesian optimization for efficient hyperparameter search."""
        
        logger.info(f"🧠 Starting Bayesian optimization...")
        logger.info(f"🎯 Target NDCG@3: {target_ndcg}")
        
        try:
            from skopt import gp_minimize
            from skopt.acquisition import gaussian_ei
            from skopt.space import Categorical, Integer, Real
        except ImportError:
            logger.warning("⚠️ scikit-optimize not available, falling back to grid search")
            return self.grid_search(train_func, data_loaders, target_ndcg, max_iterations)
        
        # Define search space for Bayesian optimization
        space = [
            Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
            Real(0.05, 0.5, name='dropout_rate'),
            Integer(256, 1024, name='hidden_dim'),
            Integer(4, 16, name='attention_heads'),
            Real(1e-5, 1e-1, name='weight_decay', prior='log-uniform'),
            Integer(16, 64, name='batch_size'),
            Integer(50, 1000, name='warmup_steps'),
            Real(0.1, 5.0, name='gradient_clip')
        ]
        
        def objective(params):
            """Objective function for optimization."""
            param_dict = {
                'learning_rate': params[0],
                'dropout_rate': params[1],
                'hidden_dim': params[2],
                'attention_heads': params[3],
                'weight_decay': params[4],
                'batch_size': params[5],
                'warmup_steps': params[6],
                'gradient_clip': params[7]
            }
            
            try:
                result = train_func(param_dict, data_loaders)
                ndcg_3 = result.get('ndcg_at_3', 0.0)
                
                # Record result
                self.results_history.append({
                    'parameters': param_dict,
                    'performance': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update best if needed
                if ndcg_3 > self.best_score:
                    self.best_score = ndcg_3
                    self.best_params = param_dict.copy()
                
                # Return negative score for minimization
                return -ndcg_3
                
            except Exception as e:
                logger.error(f"❌ Evaluation failed: {e}")
                return 0.0  # Worst possible score
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=max_iterations,
            acq_func=gaussian_ei,
            n_initial_points=5,
            random_state=42
        )
        
        target_achieved = self.best_score >= target_ndcg
        
        final_results = {
            'best_parameters': self.best_params,
            'best_performance': {
                'ndcg_at_3': self.best_score,
                'target_achieved': target_achieved
            },
            'optimization_summary': {
                'method': 'bayesian_optimization',
                'total_iterations': len(self.results_history),
                'target_ndcg': target_ndcg,
                'improvement_achieved': self.best_score - 0.60
            },
            'all_trials': self.results_history
        }
        
        self._save_tuning_results(final_results)
        
        logger.info(f"🎉 Bayesian optimization complete!")
        logger.info(f"🏆 Best NDCG@3: {self.best_score:.4f}")
        
        return final_results
    
    def progressive_search(self, train_func, data_loaders: Dict,
                          target_ndcg: float = 0.70) -> Dict:
        """Progressive search starting from best known configuration."""
        
        logger.info(f"🔄 Starting progressive hyperparameter search...")
        
        # Start with best known configuration
        base_config = {
            'learning_rate': 5e-4,
            'dropout_rate': 0.2,
            'hidden_dim': 768,
            'attention_heads': 12,
            'weight_decay': 5e-3,
            'batch_size': 16,
            'warmup_steps': 200,
            'gradient_clip': 1.0
        }
        
        # Test base configuration
        logger.info("🧪 Testing base configuration...")
        base_result = train_func(base_config, data_loaders)
        base_ndcg = base_result.get('ndcg_at_3', 0.0)
        
        self.best_score = base_ndcg
        self.best_params = base_config.copy()
        
        logger.info(f"📊 Base NDCG@3: {base_ndcg:.4f}")
        
        # Progressive refinement
        refinement_steps = [
            # Step 1: Optimize learning rate
            {'learning_rate': [3e-4, 4e-4, 5e-4, 6e-4, 7e-4]},
            # Step 2: Optimize regularization
            {'dropout_rate': [0.15, 0.2, 0.25], 'weight_decay': [3e-3, 5e-3, 7e-3]},
            # Step 3: Optimize architecture
            {'hidden_dim': [512, 768, 1024], 'attention_heads': [8, 12, 16]},
            # Step 4: Optimize training dynamics
            {'batch_size': [16, 24, 32], 'warmup_steps': [150, 200, 300]},
        ]
        
        for step_idx, param_variations in enumerate(refinement_steps):
            logger.info(f"\n🔍 Refinement step {step_idx + 1}: {list(param_variations.keys())}")
            
            step_best_score = self.best_score
            step_best_params = self.best_params.copy()
            
            # Generate combinations for this step
            param_names = list(param_variations.keys())
            param_values = list(param_variations.values())
            combinations = list(itertools.product(*param_values))
            
            for combination in combinations:
                # Create test configuration
                test_config = self.best_params.copy()
                for param_name, param_value in zip(param_names, combination):
                    test_config[param_name] = param_value
                
                try:
                    result = train_func(test_config, data_loaders)
                    ndcg_3 = result.get('ndcg_at_3', 0.0)
                    
                    logger.info(f"📊 Test config NDCG@3: {ndcg_3:.4f}")
                    
                    if ndcg_3 > step_best_score:
                        step_best_score = ndcg_3
                        step_best_params = test_config.copy()
                        logger.info(f"🏆 Step improvement: {ndcg_3:.4f}")
                
                except Exception as e:
                    logger.error(f"❌ Test failed: {e}")
                    continue
            
            # Update best configuration for next step
            if step_best_score > self.best_score:
                improvement = step_best_score - self.best_score
                logger.info(f"✅ Step {step_idx + 1} improved by {improvement:.4f}")
                self.best_score = step_best_score
                self.best_params = step_best_params.copy()
            else:
                logger.info(f"➡️ Step {step_idx + 1} - no improvement")
        
        target_achieved = self.best_score >= target_ndcg
        
        final_results = {
            'best_parameters': self.best_params,
            'best_performance': {
                'ndcg_at_3': self.best_score,
                'target_achieved': target_achieved
            },
            'optimization_summary': {
                'method': 'progressive_search',
                'base_score': base_ndcg,
                'final_score': self.best_score,
                'total_improvement': self.best_score - base_ndcg,
                'target_ndcg': target_ndcg
            }
        }
        
        self._save_tuning_results(final_results)
        
        logger.info(f"\n🎉 Progressive search complete!")
        logger.info(f"🏆 Best NDCG@3: {self.best_score:.4f}")
        logger.info(f"📈 Total improvement: {self.best_score - base_ndcg:.4f}")
        
        return final_results
    
    def _save_tuning_results(self, results: Dict):
        """Save tuning results to file."""
        
        output_dir = Path("outputs/DL/hyperparameter_tuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparameter_tuning_results_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Tuning results saved to: {filepath}")

def create_training_function(model_class, loss_function, evaluator):
    """Create a training function for hyperparameter tuning."""
    
    def train_with_params(params: Dict, data_loaders: Dict) -> Dict:
        """Train model with given parameters and return performance metrics."""
        
        # Initialize model with parameters
        model = model_class(
            hidden_dim=params.get('hidden_dim', 768),
            attention_heads=params.get('attention_heads', 12),
            dropout_rate=params.get('dropout_rate', 0.2)
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get('learning_rate', 5e-4),
            weight_decay=params.get('weight_decay', 5e-3)
        )
        
        # Training configuration
        num_epochs = 10  # Shorter for hyperparameter search
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in data_loaders['train']:
                optimizer.zero_grad()
                
                # Forward pass
                if len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                else:
                    input_ids, attention_mask, labels = batch[:3]
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_function(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    params.get('gradient_clip', 1.0)
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Early stopping for efficiency
            if epoch >= 3 and avg_loss < 0.1:
                break
        
        # Evaluation
        model.eval()
        evaluation_results = evaluator.evaluate_model(model, data_loaders['validation'])
        
        return evaluation_results
    
    return train_with_params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    tuner = HyperparameterTuner()
    logger.info("🔧 Hyperparameter tuning system initialized")
    logger.info("📊 Ready for optimization with target NDCG@3 ≥ 70%")