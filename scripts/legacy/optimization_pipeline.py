"""
Comprehensive Optimization Pipeline for 70%+ NDCG@3 Achievement
Integrates graded relevance, threshold optimization, and hyperparameter tuning.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append('src')

from dl.advanced_training import AdvancedNeuralTrainer
from dl.enhanced_neural_preprocessing import EnhancedNeuralPreprocessor
from dl.graded_relevance_enhancement import GradedRelevanceEnhancer
from dl.hyperparameter_tuning import HyperparameterTuner
from dl.improved_model_architecture import LightweightRankingModel
from dl.threshold_optimization import AdvancedThresholdOptimizer

logger = logging.getLogger(__name__)

class ComprehensiveOptimizationPipeline:
    """Complete optimization pipeline for achieving 70%+ NDCG@3."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.results = {}
        
        # Initialize components
        self.relevance_enhancer = GradedRelevanceEnhancer()
        self.threshold_optimizer = AdvancedThresholdOptimizer()
        self.hyperparameter_tuner = HyperparameterTuner()
        
        # Target performance
        self.target_ndcg = 0.70
        
    def run_complete_optimization(self) -> dict:
        """Run the complete optimization pipeline."""
        
        logger.info("🚀 Starting Comprehensive Optimization Pipeline")
        logger.info("🎯 Target: NDCG@3 ≥ 70%")
        logger.info("=" * 60)
        
        try:
            # Step 1: Enhanced Graded Relevance
            logger.info("\n📊 Step 1: Enhanced Graded Relevance Implementation")
            graded_results = self._implement_enhanced_graded_relevance()
            self.results['graded_relevance'] = graded_results
            
            # Step 2: Hyperparameter Optimization
            logger.info("\n🔧 Step 2: Hyperparameter Optimization")
            hyperparameter_results = self._optimize_hyperparameters()
            self.results['hyperparameters'] = hyperparameter_results
            
            # Step 3: Threshold Optimization
            logger.info("\n🎯 Step 3: Threshold Optimization")
            threshold_results = self._optimize_thresholds()
            self.results['thresholds'] = threshold_results
            
            # Step 4: Final Model Training with Optimal Settings
            logger.info("\n🏆 Step 4: Final Optimized Training")
            final_results = self._train_final_optimized_model()
            self.results['final_model'] = final_results
            
            # Compile comprehensive results
            final_performance = self._compile_final_results()
            
            return final_performance
            
        except Exception as e:
            logger.error(f"❌ Optimization pipeline failed: {e}")
            raise
    
    def _implement_enhanced_graded_relevance(self) -> dict:
        """Implement enhanced graded relevance with semantic boosting."""
        
        logger.info("🎯 Implementing enhanced 4-level graded relevance...")
        
        # Enhanced thresholds for better performance
        enhanced_thresholds = {
            'highly_relevant': 0.85,    # 1.0 - Very strict for highest relevance
            'relevant': 0.65,           # 0.7 - Good quality matches
            'somewhat_relevant': 0.40,  # 0.3 - Partial matches with potential
            'irrelevant': 0.0          # 0.0 - No relevance
        }
        
        enhancer = GradedRelevanceEnhancer(thresholds=enhanced_thresholds)
        
        # Create advanced graded data with semantic enhancement
        input_file = "data/processed/enhanced_training_data.json"
        output_file = "data/processed/enhanced_training_data_optimized.json"
        
        if not Path(input_file).exists():
            logger.error(f"❌ Training data not found: {input_file}")
            return {'error': 'Training data not found'}
        
        stats = enhancer.create_advanced_graded_data(
            input_file, output_file, semantic_boost=True
        )
        
        logger.info("✅ Enhanced graded relevance implemented!")
        logger.info(f"📊 Relevance distribution: {stats['relevance_distribution']}")
        logger.info(f"🔧 Semantic enhancements: {stats['semantic_enhancements']}")
        
        return {
            'status': 'completed',
            'enhanced_file': output_file,
            'statistics': stats,
            'thresholds_used': enhanced_thresholds
        }
    
    def _optimize_hyperparameters(self) -> dict:
        """Optimize hyperparameters for maximum performance."""
        
        logger.info("🔍 Starting hyperparameter optimization...")
        
        # Load optimized training data
        data_file = "data/processed/enhanced_training_data_optimized.json"
        if not Path(data_file).exists():
            data_file = "data/processed/enhanced_training_data_graded.json"
        
        # Create training function
        def quick_train_and_evaluate(params: dict, data_loaders=None) -> dict:
            """Quick training function for hyperparameter optimization."""
            
            try:
                # Initialize components
                preprocessor = EnhancedNeuralPreprocessor(config={})
                
                # Load and prepare data
                train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
                    batch_size=params.get('batch_size', 16)
                )
                
                # Initialize model with parameters
                model = LightweightRankingModel(
                    embedding_dim=256,
                    hidden_dim=params.get('hidden_dim', 768),
                    dropout_rate=params.get('dropout_rate', 0.2),
                    num_attention_heads=params.get('attention_heads', 12)
                )
                
                # Initialize trainer
                trainer = AdvancedNeuralTrainer(
                    model=model,
                    config={
                        'learning_rate': params.get('learning_rate', 5e-4),
                        'weight_decay': params.get('weight_decay', 5e-3),
                        'epochs': 8,  # Shorter for optimization
                        'patience': 4,
                        'device': str(self.device)
                    }
                )
                
                # Train model
                training_results = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    save_best=False  # Don't save during optimization
                )
                
                # Quick evaluation
                model.eval()
                evaluation_results = trainer.evaluate_model(
                    model=model,
                    test_loader=val_loader,  # Use validation for speed
                    calculate_ndcg=True
                )
                
                return evaluation_results
                
            except Exception as e:
                logger.error(f"❌ Quick training failed: {e}")
                return {'ndcg_at_3': 0.0, 'accuracy': 0.0, 'f1_score': 0.0}
        
        # Run progressive search (most efficient for our case)
        hyperparameter_results = self.hyperparameter_tuner.progressive_search(
            train_func=quick_train_and_evaluate,
            data_loaders={},  # Will be created in function
            target_ndcg=self.target_ndcg
        )
        
        logger.info("✅ Hyperparameter optimization completed!")
        best_params = hyperparameter_results['best_parameters']
        best_score = hyperparameter_results['best_performance']['ndcg_at_3']
        logger.info(f"🏆 Best NDCG@3: {best_score:.4f}")
        logger.info(f"📋 Best parameters: {best_params}")
        
        return hyperparameter_results
    
    def _optimize_thresholds(self) -> dict:
        """Optimize classification thresholds."""
        
        logger.info("🎯 Optimizing classification thresholds...")
        
        # Get best hyperparameters from previous step
        best_params = self.results['hyperparameters']['best_parameters']
        
        # Load data with best batch size
        data_file = "data/processed/enhanced_training_data_optimized.json"
        if not Path(data_file).exists():
            data_file = "data/processed/enhanced_training_data_graded.json"
        
        preprocessor = EnhancedNeuralPreprocessor(config={})
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
            batch_size=best_params.get('batch_size', 16)
        )
        
        # Train model with best hyperparameters
        model = LightweightRankingModel(
            embedding_dim=256,
            hidden_dim=best_params.get('hidden_dim', 768),
            dropout_rate=best_params.get('dropout_rate', 0.2),
            num_attention_heads=best_params.get('attention_heads', 12)
        )
        
        trainer = AdvancedNeuralTrainer(
            model=model,
            config={
                'learning_rate': best_params.get('learning_rate', 5e-4),
                'weight_decay': best_params.get('weight_decay', 5e-3),
                'epochs': 12,
                'patience': 5,
                'device': str(self.device)
            }
        )
        
        # Train the model
        logger.info("🏋️ Training model for threshold optimization...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_best=True
        )
        
        # Optimize thresholds
        threshold_results = self.threshold_optimizer.optimize_graded_thresholds(
            model=model,
            val_loader=val_loader,
            target_ndcg=self.target_ndcg
        )
        
        logger.info("✅ Threshold optimization completed!")
        best_thresholds = threshold_results['best_thresholds']
        threshold_ndcg = threshold_results['best_ndcg_3']
        logger.info(f"🏆 Optimized NDCG@3: {threshold_ndcg:.4f}")
        logger.info(f"🎯 Optimal thresholds: {best_thresholds}")
        
        return threshold_results
    
    def _train_final_optimized_model(self) -> dict:
        """Train final model with all optimizations."""
        
        logger.info("🏆 Training final optimized model...")
        
        # Get optimal settings from previous steps
        best_params = self.results['hyperparameters']['best_parameters']
        best_thresholds = self.results['thresholds']['best_thresholds']
        
        # Load data
        data_file = "data/processed/enhanced_training_data_optimized.json"
        if not Path(data_file).exists():
            data_file = "data/processed/enhanced_training_data_graded.json"
        
        preprocessor = EnhancedNeuralPreprocessor(config={})
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
            batch_size=best_params.get('batch_size', 16)
        )
        
        # Initialize optimized model
        final_model = LightweightRankingModel(
            embedding_dim=256,
            hidden_dim=best_params.get('hidden_dim', 768),
            dropout_rate=best_params.get('dropout_rate', 0.2),
            num_attention_heads=best_params.get('attention_heads', 12)
        )
        
        # Initialize trainer with optimal settings
        final_trainer = AdvancedNeuralTrainer(
            model=final_model,
            config={
                'learning_rate': best_params.get('learning_rate', 5e-4),
                'weight_decay': best_params.get('weight_decay', 5e-3),
                'epochs': 20,  # Full training for final model
                'patience': 8,
                'device': str(self.device),
                'use_graded_thresholds': True,
                'graded_thresholds': best_thresholds
            }
        )
        
        # Final training
        logger.info("🚀 Starting final optimized training...")
        final_training_results = final_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_best=True
        )
        
        # Comprehensive evaluation
        logger.info("📊 Performing comprehensive evaluation...")
        final_evaluation = final_trainer.evaluate_model(
            model=final_model,
            test_loader=test_loader,
            calculate_ndcg=True,
            detailed_analysis=True
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/dl/optimized_model_{timestamp}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'hyperparameters': best_params,
            'thresholds': best_thresholds,
            'performance': final_evaluation,
            'training_results': final_training_results
        }, model_path)
        
        logger.info(f"💾 Final optimized model saved: {model_path}")
        
        return {
            'model_path': model_path,
            'training_results': final_training_results,
            'evaluation_results': final_evaluation,
            'hyperparameters_used': best_params,
            'thresholds_used': best_thresholds
        }
    
    def _compile_final_results(self) -> dict:
        """Compile and analyze final optimization results."""
        
        logger.info("\n📋 Compiling final optimization results...")
        
        # Extract key metrics
        final_evaluation = self.results['final_model']['evaluation_results']
        final_ndcg = final_evaluation.get('ndcg_at_3', 0.0)
        final_accuracy = final_evaluation.get('accuracy', 0.0)
        final_f1 = final_evaluation.get('f1_score', 0.0)
        
        # Check target achievement
        target_achieved = final_ndcg >= self.target_ndcg
        improvement_from_baseline = final_ndcg - 0.60  # From baseline 60.2%
        
        comprehensive_results = {
            'optimization_summary': {
                'target_ndcg': self.target_ndcg,
                'achieved_ndcg': final_ndcg,
                'target_achieved': target_achieved,
                'improvement_from_baseline': improvement_from_baseline,
                'final_accuracy': final_accuracy,
                'final_f1_score': final_f1
            },
            'optimization_steps': {
                'graded_relevance': self.results['graded_relevance']['status'],
                'hyperparameter_optimization': self.results['hyperparameters']['best_performance'],
                'threshold_optimization': self.results['thresholds']['best_ndcg_3'],
                'final_training': 'completed'
            },
            'best_configuration': {
                'hyperparameters': self.results['hyperparameters']['best_parameters'],
                'thresholds': self.results['thresholds']['best_thresholds'],
                'graded_relevance_thresholds': self.results['graded_relevance']['thresholds_used']
            },
            'detailed_results': self.results
        }
        
        # Save comprehensive results
        output_path = f"outputs/DL/comprehensive_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Final summary
        logger.info("\n🎉 COMPREHENSIVE OPTIMIZATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"🎯 Target NDCG@3: {self.target_ndcg:.1%}")
        logger.info(f"🏆 Achieved NDCG@3: {final_ndcg:.1%}")
        logger.info(f"✅ Target Achieved: {'YES' if target_achieved else 'NO'}")
        logger.info(f"📈 Improvement from Baseline: +{improvement_from_baseline:.1%}")
        logger.info(f"🎯 Final Accuracy: {final_accuracy:.1%}")
        logger.info(f"📊 Final F1 Score: {final_f1:.3f}")
        logger.info(f"💾 Results saved: {output_path}")
        
        if target_achieved:
            logger.info("🎊 CONGRATULATIONS! 70%+ NDCG@3 TARGET ACHIEVED!")
        else:
            gap = self.target_ndcg - final_ndcg
            logger.info(f"📏 Gap to target: {gap:.1%}")
            logger.info("💡 Consider additional optimization rounds")
        
        return comprehensive_results

def run_optimization_pipeline():
    """Run the complete optimization pipeline."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pipeline = ComprehensiveOptimizationPipeline()
    results = pipeline.run_complete_optimization()
    
    return results

if __name__ == "__main__":
    results = run_optimization_pipeline()