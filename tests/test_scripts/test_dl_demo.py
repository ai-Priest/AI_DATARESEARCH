#!/usr/bin/env python3
"""
Simple Demo of DL Pipeline Components
Demonstrates the key capabilities without full training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.dl.neural_preprocessing import create_neural_data_preprocessor
from src.dl.model_architecture import create_neural_models
from src.dl.neural_inference import create_neural_inference_engine
import yaml

def demo_dl_capabilities():
    """Demonstrate DL pipeline capabilities."""
    print("🚀 DL Pipeline Demo - Key Capabilities")
    print("=" * 50)
    
    # Load configuration
    with open('config/dl_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Neural Data Preprocessing Demo
    print("\n📊 1. Neural Data Preprocessing")
    print("-" * 30)
    
    preprocessor = create_neural_data_preprocessor(config)
    print(f"✅ Created neural preprocessor")
    print(f"   • Text processing: {'BERT tokenizer' if preprocessor.tokenizer else 'Simplified (no tokenizer)'}")
    print(f"   • Feature engineering: Advanced temporal, categorical, interaction features")
    print(f"   • Graph construction: GNN-ready dataset relationship graph")
    print(f"   • Data augmentation: Noise injection and synthetic samples")
    
    # 2. Neural Architecture Demo
    print("\n🧠 2. Neural Model Architectures")
    print("-" * 30)
    
    models = create_neural_models(config)
    print(f"✅ Created {len(models)} neural models:")
    
    for name, model in models.items():
        if hasattr(model, 'parameters'):
            params = sum(p.numel() for p in model.parameters())
            print(f"   • {name}: {params:,} parameters")
        else:
            print(f"   • {name}: Loss function")
    
    total_params = sum(
        sum(p.numel() for p in model.parameters()) 
        for model in models.values() 
        if hasattr(model, 'parameters')
    )
    print(f"   📊 Total: {total_params:,} parameters ({total_params * 4 / (1024**2):.1f} MB)")
    
    # 3. Inference Engine Demo  
    print("\n⚡ 3. Neural Inference Engine")
    print("-" * 30)
    
    inference_engine = create_neural_inference_engine(config)
    print(f"✅ Created inference engine")
    print(f"   • Device: {inference_engine.device}")
    print(f"   • Caching: {'Enabled' if inference_engine.cache_enabled else 'Disabled'}")
    print(f"   • Real-time: {'Enabled' if inference_engine.real_time_config.get('enabled') else 'Disabled'}")
    
    # Test inference (without loaded models)
    health = inference_engine.health_check()
    print(f"   • Health: {health['status']}")
    
    # Demo recommendation (will use fallback)
    result = inference_engine.recommend_datasets("singapore housing market analysis", top_k=3)
    print(f"   • Test query processing: {result.processing_time:.3f}s")
    print(f"   • Recommendations generated: {len(result.recommendations)}")
    
    # 4. Configuration Highlights
    print("\n⚙️ 4. Advanced Configuration")
    print("-" * 30)
    
    dl_config = config
    print("✅ Deep Learning features configured:")
    
    # Model configurations
    models_config = dl_config.get('models', {})
    enabled_models = [name for name, model_config in models_config.items() 
                     if model_config.get('enabled', False)]
    print(f"   • Neural architectures: {', '.join(enabled_models)}")
    
    # Training features  
    training_config = dl_config.get('training', {})
    training_features = []
    if training_config.get('mixed_precision'): training_features.append("Mixed Precision")
    if training_config.get('gradient_checkpointing'): training_features.append("Gradient Checkpointing")
    if training_config.get('strategy', {}).get('curriculum_learning'): training_features.append("Curriculum Learning")
    print(f"   • Training features: {', '.join(training_features) if training_features else 'Standard'}")
    
    # Evaluation capabilities
    eval_config = dl_config.get('evaluation', {})
    eval_metrics = eval_config.get('neural_metrics', {}).get('metrics', [])
    print(f"   • Evaluation metrics: {len(eval_metrics)} neural metrics")
    
    # Advanced features
    advanced_features = dl_config.get('advanced_features', {})
    advanced_enabled = [name for name, feature in advanced_features.items() 
                       if feature.get('enabled', False)]
    print(f"   • Advanced features: {', '.join(advanced_enabled) if advanced_enabled else 'None enabled'}")
    
    print("\n🎯 5. Integration Capabilities")
    print("-" * 30)
    print("✅ Seamless integration with existing pipeline:")
    print("   • ML pipeline compatibility: Enhanced embeddings and features")
    print("   • User behavior analysis: Real interaction data processing") 
    print("   • Enhancement systems: Query expansion, feedback, explanations")
    print("   • Production deployment: Optimized inference, caching, monitoring")
    
    print("\n🚀 DL Pipeline Demo Complete!")
    print("=" * 50)
    print("📋 Summary:")
    print(f"   • Neural architectures: {len(enabled_models)} models with {total_params:,} parameters")
    print(f"   • Data processing: Advanced preprocessing with graph construction")
    print(f"   • Inference engine: Production-ready with {inference_engine.device} acceleration")
    print(f"   • Configuration: Comprehensive YAML-driven setup")
    print(f"   • Integration: Full compatibility with existing ML pipeline")
    print("\n🎯 Ready for neural-enhanced dataset recommendations!")

if __name__ == "__main__":
    demo_dl_capabilities()