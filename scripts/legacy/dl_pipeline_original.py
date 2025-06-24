#!/usr/bin/env python3
"""
Deep Learning Pipeline - Advanced Neural Network Training and Evaluation
Main orchestrator for the DL phase: neural preprocessing, model training, evaluation, and inference.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dl.neural_preprocessing import create_neural_data_preprocessor
from src.dl.model_architecture import create_neural_models
from src.dl.advanced_training import create_advanced_trainer
from src.dl.deep_evaluation import create_deep_evaluator
from src.dl.neural_inference import create_neural_inference_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/dl_pipeline.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class DLPipeline:
    """Deep Learning Pipeline orchestrator for neural dataset recommendation."""
    
    def __init__(self, config_path: str = "config/dl_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Pipeline state
        self.pipeline_state = {
            'preprocessing_completed': False,
            'training_completed': False,
            'evaluation_completed': False,
            'inference_ready': False
        }
        
        # Results storage
        self.results = {
            'preprocessing': None,
            'training': None,
            'evaluation': None,
            'inference_engine': None
        }
        
        # Setup output directories
        self._setup_directories()
        
        logger.info(f"🚀 DL Pipeline initialized - Execution ID: {self.execution_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise
    
    def _setup_directories(self):
        """Setup required directories for DL pipeline."""
        directories = [
            self.config.get('outputs', {}).get('models_dir', 'models/dl'),
            self.config.get('outputs', {}).get('logs_dir', 'logs/dl'),
            self.config.get('outputs', {}).get('reports_dir', 'outputs/DL/reports'),
            self.config.get('outputs', {}).get('visualizations_dir', 'outputs/DL/visualizations'),
            self.config.get('outputs', {}).get('evaluations_dir', 'outputs/DL/evaluations')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Output directories created")
    
    def run_complete_pipeline(self, 
                            skip_preprocessing: bool = False,
                            skip_training: bool = False,
                            skip_evaluation: bool = False) -> Dict[str, Any]:
        """Execute the complete DL pipeline."""
        logger.info("🎯 Starting complete DL pipeline execution")
        start_time = time.time()
        
        try:
            # Phase 1: Neural Data Preprocessing
            if not skip_preprocessing:
                logger.info("📊 Phase 1: Neural Data Preprocessing")
                preprocessing_results = self.run_neural_preprocessing()
                self.results['preprocessing'] = preprocessing_results
                self.pipeline_state['preprocessing_completed'] = True
                logger.info("✅ Neural preprocessing completed")
            
            # Phase 2: Advanced Neural Training
            if not skip_training:
                logger.info("🧠 Phase 2: Advanced Neural Training")
                training_results = self.run_neural_training()
                self.results['training'] = training_results
                self.pipeline_state['training_completed'] = True
                logger.info("✅ Neural training completed")
            
            # Phase 3: Deep Evaluation
            if not skip_evaluation:
                logger.info("🎯 Phase 3: Deep Evaluation")
                evaluation_results = self.run_deep_evaluation()
                self.results['evaluation'] = evaluation_results
                self.pipeline_state['evaluation_completed'] = True
                logger.info("✅ Deep evaluation completed")
            
            # Phase 4: Inference Engine Setup
            logger.info("⚡ Phase 4: Neural Inference Engine")
            inference_engine = self.setup_inference_engine()
            self.results['inference_engine'] = inference_engine
            self.pipeline_state['inference_ready'] = True
            logger.info("✅ Inference engine ready")
            
            # Generate final report
            final_results = self._generate_final_report()
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Complete DL pipeline executed successfully in {total_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ DL pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_neural_preprocessing(self) -> Dict[str, Any]:
        """Execute neural data preprocessing phase."""
        logger.info("🧠 Executing neural data preprocessing")
        
        try:
            # Create preprocessor
            preprocessor = create_neural_data_preprocessor(self.config)
            
            # Run preprocessing pipeline
            processed_data, validation_results = preprocessor.process_complete_pipeline()
            
            # Log preprocessing summary
            metadata = processed_data.get('metadata', {})
            logger.info(f"📊 Preprocessing Summary:")
            logger.info(f"  • Total datasets: {metadata.get('num_datasets', 0)}")
            logger.info(f"  • Unique users: {metadata.get('num_users', 0)}")
            logger.info(f"  • Text vocab size: {metadata.get('text_vocab_size', 0)}")
            logger.info(f"  • Feature dimensions: {metadata.get('feature_dimensions', {})}")
            
            # Validation status
            validation_status = validation_results.get('status', 'UNKNOWN')
            logger.info(f"  • Validation status: {validation_status}")
            
            if validation_results.get('warnings'):
                for warning in validation_results['warnings']:
                    logger.warning(f"⚠️ {warning}")
            
            return {
                'processed_data': processed_data,
                'validation_results': validation_results,
                'preprocessing_metadata': {
                    'execution_time': time.time(),
                    'config_used': self.config.get('data_processing', {}),
                    'validation_status': validation_status
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Neural preprocessing failed: {e}")
            raise
    
    def run_neural_training(self) -> Dict[str, Any]:
        """Execute advanced neural training phase."""
        logger.info("🚀 Executing advanced neural training")
        
        try:
            # Get preprocessed data
            if self.results['preprocessing'] is None:
                logger.info("📊 Running preprocessing as prerequisite")
                preprocessing_results = self.run_neural_preprocessing()
                self.results['preprocessing'] = preprocessing_results
            
            processed_data = self.results['preprocessing']['processed_data']
            
            # Create trainer
            trainer = create_advanced_trainer(self.config)
            
            # Execute training pipeline
            training_results = trainer.train_complete_pipeline(processed_data)
            
            # Log training summary
            logger.info(f"🎯 Training Summary:")
            logger.info(f"  • Models trained: {len(trainer.models)}")
            logger.info(f"  • Best epoch: {training_results.get('best_epoch', 'N/A')}")
            logger.info(f"  • Best metric: {training_results.get('best_metric', 0):.4f}")
            logger.info(f"  • Final train loss: {training_results['training_results'].get('final_train_loss', 0):.4f}")
            logger.info(f"  • Final val loss: {training_results['training_results'].get('final_val_loss', 0):.4f}")
            
            return {
                'training_results': training_results,
                'trained_models': trainer.models,
                'training_history': trainer.training_history,
                'training_metadata': {
                    'device_used': str(trainer.device),
                    'mixed_precision': trainer.mixed_precision,
                    'total_parameters': sum(
                        sum(p.numel() for p in model.parameters()) 
                        for model in trainer.models.values() 
                        if hasattr(model, 'parameters')
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Neural training failed: {e}")
            raise
    
    def run_deep_evaluation(self) -> Dict[str, Any]:
        """Execute deep evaluation phase."""
        logger.info("📊 Executing deep evaluation")
        
        try:
            # Get required data
            if self.results['training'] is None:
                logger.info("🧠 Running training as prerequisite")
                self.run_neural_training()
            
            if self.results['preprocessing'] is None:
                logger.info("📊 Running preprocessing as prerequisite")
                self.run_neural_preprocessing()
            
            trained_models = self.results['training']['trained_models']
            processed_data = self.results['preprocessing']['processed_data']
            test_data = processed_data.get('test', {})
            
            # Load user behavior data if available
            user_behavior_data = None
            try:
                import pandas as pd
                user_behavior_path = self.config.get('data_processing', {}).get('input_paths', {}).get('user_behavior', 'data/raw/user_behaviour.csv')
                if Path(user_behavior_path).exists():
                    user_behavior_data = pd.read_csv(user_behavior_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not load user behavior data: {e}")
            
            # Create evaluator
            evaluator = create_deep_evaluator(self.config)
            
            # Execute evaluation pipeline
            evaluation_results = evaluator.evaluate_complete_pipeline(
                trained_models, 
                test_data, 
                user_behavior_data
            )
            
            # Log evaluation summary
            summary_metrics = evaluation_results.get('summary_metrics', {})
            logger.info(f"🎯 Evaluation Summary:")
            logger.info(f"  • Average NDCG@3: {summary_metrics.get('avg_ndcg_at_3', 0):.4f}")
            logger.info(f"  • Average F1 Score: {summary_metrics.get('avg_f1_score', 0):.4f}")
            logger.info(f"  • Best Model: {summary_metrics.get('best_model', 'N/A')}")
            logger.info(f"  • Test Set Size: {evaluation_results.get('evaluation_metadata', {}).get('test_set_size', 0)}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Deep evaluation failed: {e}")
            raise
    
    def setup_inference_engine(self) -> Any:
        """Setup neural inference engine."""
        logger.info("⚡ Setting up neural inference engine")
        
        try:
            # Create inference engine
            inference_engine = create_neural_inference_engine(self.config)
            
            # Load trained models
            models_dir = self.config.get('outputs', {}).get('models_dir', 'models/dl')
            loading_results = inference_engine.load_models(models_dir)
            
            # Optimize for production
            inference_engine.optimize_for_production()
            
            # Perform health check
            health_status = inference_engine.health_check()
            
            # Log setup summary
            logger.info(f"⚡ Inference Engine Summary:")
            logger.info(f"  • Models loaded: {len(loading_results)}")
            logger.info(f"  • Health status: {health_status['status']}")
            logger.info(f"  • Active model: {health_status.get('active_model', 'N/A')}")
            logger.info(f"  • Device: {health_status.get('device', 'N/A')}")
            
            # Test inference
            if health_status['status'] == 'healthy':
                test_result = inference_engine.recommend_datasets(
                    "singapore housing market analysis", 
                    top_k=3
                )
                logger.info(f"  • Test inference: {len(test_result.recommendations)} recommendations in {test_result.processing_time:.3f}s")
            
            return inference_engine
            
        except Exception as e:
            logger.error(f"❌ Inference engine setup failed: {e}")
            raise
    
    def run_preprocessing_only(self) -> Dict[str, Any]:
        """Run only neural preprocessing phase."""
        logger.info("📊 Running neural preprocessing only")
        return self.run_neural_preprocessing()
    
    def run_training_only(self) -> Dict[str, Any]:
        """Run only neural training phase."""
        logger.info("🧠 Running neural training only")
        return self.run_neural_training()
    
    def run_evaluation_only(self) -> Dict[str, Any]:
        """Run only deep evaluation phase."""
        logger.info("🎯 Running deep evaluation only")
        return self.run_deep_evaluation()
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("📝 Generating final DL pipeline report")
        
        final_results = {
            'execution_metadata': {
                'execution_id': self.execution_id,
                'timestamp': datetime.now().isoformat(),
                'config_path': str(self.config_path),
                'pipeline_state': self.pipeline_state
            },
            'phases': {
                'preprocessing': self.results.get('preprocessing'),
                'training': self.results.get('training'),
                'evaluation': self.results.get('evaluation'),
                'inference': {
                    'engine_ready': self.pipeline_state.get('inference_ready', False),
                    'health_status': self.results['inference_engine'].health_check() if self.results.get('inference_engine') else None
                }
            },
            'summary': self._create_execution_summary()
        }
        
        # Save final report
        report_path = Path(self.config.get('outputs', {}).get('reports_dir', 'outputs/DL/reports')) / f'dl_pipeline_report_{self.execution_id}.json'
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(self._make_json_serializable(final_results), f, indent=2)
            logger.info(f"📄 Final report saved to {report_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save final report: {e}")
        
        return final_results
    
    def _create_execution_summary(self) -> Dict[str, Any]:
        """Create execution summary."""
        summary = {
            'phases_completed': sum(self.pipeline_state.values()),
            'total_phases': len(self.pipeline_state),
            'success_rate': sum(self.pipeline_state.values()) / len(self.pipeline_state),
            'key_metrics': {}
        }
        
        # Add key metrics from each phase
        if self.results.get('evaluation'):
            summary['key_metrics'].update(
                self.results['evaluation'].get('summary_metrics', {})
            )
        
        if self.results.get('training'):
            summary['key_metrics']['best_training_metric'] = self.results['training'].get('training_results', {}).get('best_metric', 0)
        
        return summary
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        import numpy as np
        import torch
        
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return "<<tensor_data>>"  # Placeholder for large tensors
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return "<<object>>"  # Placeholder for complex objects
        else:
            return obj


def main():
    """Main entry point for DL pipeline."""
    parser = argparse.ArgumentParser(description="Deep Learning Pipeline for Dataset Recommendation")
    
    # Pipeline execution options
    parser.add_argument('--config', '-c', 
                       default='config/dl_config.yml',
                       help='Path to DL configuration file')
    
    parser.add_argument('--preprocess-only', 
                       action='store_true',
                       help='Run only neural preprocessing phase')
    
    parser.add_argument('--train-only', 
                       action='store_true',
                       help='Run only neural training phase')
    
    parser.add_argument('--evaluate-only', 
                       action='store_true',
                       help='Run only deep evaluation phase')
    
    parser.add_argument('--skip-preprocessing', 
                       action='store_true',
                       help='Skip neural preprocessing phase')
    
    parser.add_argument('--skip-training', 
                       action='store_true',
                       help='Skip neural training phase')
    
    parser.add_argument('--skip-evaluation', 
                       action='store_true',
                       help='Skip deep evaluation phase')
    
    parser.add_argument('--validate-only', 
                       action='store_true',
                       help='Validate configuration without execution')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = DLPipeline(config_path=args.config)
        
        # Validate configuration
        if args.validate_only:
            logger.info("✅ Configuration validation successful")
            return
        
        # Execute based on arguments
        if args.preprocess_only:
            results = pipeline.run_preprocessing_only()
        elif args.train_only:
            results = pipeline.run_training_only()
        elif args.evaluate_only:
            results = pipeline.run_evaluation_only()
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                skip_preprocessing=args.skip_preprocessing,
                skip_training=args.skip_training,
                skip_evaluation=args.skip_evaluation
            )
        
        logger.info("🎉 DL Pipeline execution completed successfully!")
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            logger.info(f"📊 Execution Summary:")
            logger.info(f"  • Phases completed: {summary['phases_completed']}/{summary['total_phases']}")
            logger.info(f"  • Success rate: {summary['success_rate']:.1%}")
            
            key_metrics = summary.get('key_metrics', {})
            if key_metrics:
                logger.info(f"  • Key metrics: {key_metrics}")
        
    except Exception as e:
        logger.error(f"❌ DL Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()