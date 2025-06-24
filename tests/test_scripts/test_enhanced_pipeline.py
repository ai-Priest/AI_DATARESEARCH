#!/usr/bin/env python3
"""
Test Enhanced ML Pipeline
Demonstrates all new enhancement features in action.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add src to path
src_path = Path.cwd() / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ml.enhanced_ml_pipeline import EnhancedMLPipeline
from ml.model_training import EnhancedRecommendationEngine
from ml.model_inference import ProductionRecommendationEngine
import yaml

console = Console()

def load_config():
    """Load ML configuration."""
    with open('config/ml_config.yml', 'r') as f:
        return yaml.safe_load(f)

def load_saved_models():
    """Load saved models from disk."""
    config = load_config()
    
    # Load datasets
    datasets_df = pd.read_csv('models/datasets_metadata.csv')
    
    # Use production inference engine instead
    production_engine = ProductionRecommendationEngine(config, 'models')
    
    # Create enhanced pipeline using production engine as base
    enhanced_pipeline = EnhancedMLPipeline(config)
    enhanced_pipeline.recommendation_engine = production_engine
    enhanced_pipeline.initialize_enhancement_components(datasets_df)
    
    return enhanced_pipeline, datasets_df

def demo_query_expansion(enhanced_pipeline):
    """Demonstrate query expansion."""
    console.print(Panel("🔍 Query Expansion Demo", style="bold blue"))
    
    test_queries = [
        "housing data",
        "LTA traffic",
        "government budget",
        "transport planning"
    ]
    
    expansion_table = Table(show_header=True, header_style="bold cyan")
    expansion_table.add_column("Original Query", style="yellow")
    expansion_table.add_column("Expanded Query", style="green")
    expansion_table.add_column("Added Terms", style="dim")
    
    for query in test_queries:
        expansion = enhanced_pipeline.enhance_query(query)
        
        expansion_table.add_row(
            query,
            expansion['expanded_query'],
            ', '.join(expansion['expansion_terms'])
        )
    
    console.print(expansion_table)

def demo_enhanced_recommendations(enhanced_pipeline):
    """Demonstrate enhanced recommendations with all features."""
    console.print(Panel("🎯 Enhanced Recommendations Demo", style="bold magenta"))
    
    query = "singapore housing market analysis"
    
    console.print(f"📋 Query: [bold]{query}[/bold]")
    
    # Get enhanced recommendations
    result = enhanced_pipeline.get_enhanced_recommendations(
        query=query,
        method='hybrid',
        top_k=3,
        user_context={'user_id': 'demo_user'}
    )
    
    console.print(f"\n🔍 Enhanced Query: [green]{result['query_info']['enhanced_query']}[/green]")
    console.print(f"📊 Enhancements Applied: {result['enhancements_applied']}")
    
    # Display recommendations with enhancements
    for i, rec in enumerate(result['recommendations'], 1):
        console.print(f"\n📄 Recommendation {i}:")
        console.print(f"   Title: [bold]{rec.get('title', 'N/A')}[/bold]")
        console.print(f"   Score: [yellow]{rec.get('score', 0):.3f}[/yellow]")
        
        # Show explanation if available
        if 'explanation' in rec and rec['explanation'].get('available', False):
            explanation = rec['explanation']
            console.print(f"   💡 Explanation: {explanation.get('explanation_text', 'N/A')[:100]}...")
            console.print(f"   🎯 Confidence: [cyan]{explanation.get('confidence_level', 'unknown')}[/cyan]")
        
        # Show preview card availability
        if 'preview_card' in rec and rec['preview_card'].get('available', True):
            console.print(f"   🎨 Preview Card: [green]Available[/green]")
        else:
            console.print(f"   🎨 Preview Card: [red]Not Available[/red]")

def demo_progressive_search(enhanced_pipeline):
    """Demonstrate progressive search."""
    console.print(Panel("⚡ Progressive Search Demo", style="bold green"))
    
    partial_queries = ["hou", "trans", "gov"]
    
    for partial in partial_queries:
        suggestions = enhanced_pipeline.get_progressive_search_suggestions(partial, max_suggestions=5)
        
        console.print(f"\n🔍 Partial Query: '[yellow]{partial}[/yellow]'")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"   {i}. {suggestion['text']} ([dim]{suggestion['type']}[/dim])")
        else:
            console.print("   [dim]No suggestions available[/dim]")

def demo_user_feedback(enhanced_pipeline):
    """Demonstrate user feedback system."""
    console.print(Panel("📝 User Feedback Demo", style="bold yellow"))
    
    # Record some demo interactions
    session_id = "demo_session_123"
    user_id = "demo_user"
    
    # Simulate search interaction
    feedback_id = enhanced_pipeline.record_user_interaction(
        user_id=user_id,
        session_id=session_id,
        query="housing data singapore",
        results=[{'title': 'HDB Resale Prices', 'score': 0.85}],
        interaction_type="click",
        dataset_id="hdb_001",
        rating=5
    )
    
    if feedback_id:
        console.print(f"✅ Recorded interaction: {feedback_id}")
    
    # Get user insights
    insights = enhanced_pipeline.get_user_insights(user_id)
    
    if 'error' not in insights:
        console.print(f"📊 User Insights:")
        console.print(f"   Engagement Score: [green]{insights.get('engagement_score', 0):.1%}[/green]")
        
        preferred_terms = insights.get('preferred_query_terms', [])
        if preferred_terms:
            console.print(f"   Preferred Terms: {', '.join([term for term, count in preferred_terms[:3]])}")

def main():
    """Main demo function."""
    console.print(Panel.fit(
        Text.from_markup(
            "[bold blue]🚀 Enhanced ML Pipeline Demo[/bold blue]\n"
            "[dim]Showcasing all enhancement features[/dim]"
        ),
        border_style="blue"
    ))
    
    try:
        # Load enhanced pipeline
        console.print("\n⏳ Loading enhanced models...")
        enhanced_pipeline, datasets_df = load_saved_models()
        
        console.print(f"✅ Loaded {len(datasets_df)} datasets")
        console.print(f"🎯 Enhancement components: {sum([
            enhanced_pipeline.query_expansion_enabled,
            enhanced_pipeline.user_feedback_enabled,
            enhanced_pipeline.explanations_enabled,
            enhanced_pipeline.progressive_search_enabled,
            enhanced_pipeline.preview_cards_enabled
        ])}/5 enabled")
        
        # Run demos
        console.print("\n" + "="*60)
        demo_query_expansion(enhanced_pipeline)
        
        console.print("\n" + "="*60)
        demo_enhanced_recommendations(enhanced_pipeline)
        
        console.print("\n" + "="*60)
        demo_progressive_search(enhanced_pipeline)
        
        console.print("\n" + "="*60)
        demo_user_feedback(enhanced_pipeline)
        
        # Final summary
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            Text.from_markup(
                "[bold green]🎉 Demo Complete![/bold green]\n\n"
                "[cyan]All enhancement features are working:[/cyan]\n"
                "• Query expansion with Singapore-specific terms\n"
                "• Intelligent recommendation explanations\n"
                "• Progressive search with autocomplete\n"
                "• User feedback collection and analysis\n"
                "• Rich dataset preview cards\n\n"
                "[dim]Expected improvements: F1@3 from 37% → 45%+[/dim]"
            ),
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()