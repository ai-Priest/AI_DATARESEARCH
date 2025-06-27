
# Model Architecture Diagram Generator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_neural_architecture_diagram():
    """Create visual diagram of GradedRankingModel architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    input_color = '#E8F4FD'
    processing_color = '#B3E5FC'
    attention_color = '#4FC3F7'
    fusion_color = '#29B6F6'
    output_color = '#0288D1'
    
    # Input layer
    query_box = FancyBboxPatch((0.5, 10), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color, 
                               edgecolor='black')
    ax.add_patch(query_box)
    ax.text(1.5, 10.5, 'Query\nEmbedding\n(768d)', ha='center', va='center', fontweight='bold')
    
    dataset_box = FancyBboxPatch((7, 10), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=input_color,
                                 edgecolor='black')
    ax.add_patch(dataset_box)
    ax.text(8, 10.5, 'Dataset\nEmbedding\n(768d)', ha='center', va='center', fontweight='bold')
    
    # Encoder layers
    query_encoder = FancyBboxPatch((0.5, 8), 2, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=processing_color,
                                   edgecolor='black')
    ax.add_patch(query_encoder)
    ax.text(1.5, 8.5, 'Query Encoder\n(128d)', ha='center', va='center', fontweight='bold')
    
    dataset_encoder = FancyBboxPatch((7, 8), 2, 1,
                                     boxstyle="round,pad=0.1", 
                                     facecolor=processing_color,
                                     edgecolor='black')
    ax.add_patch(dataset_encoder)
    ax.text(8, 8.5, 'Dataset Encoder\n(128d)', ha='center', va='center', fontweight='bold')
    
    # Cross-attention
    attention_box = FancyBboxPatch((3.5, 6), 3, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=attention_color,
                                   edgecolor='black')
    ax.add_patch(attention_box)
    ax.text(5, 6.75, 'Cross-Attention\n(4 heads, 128d)\nLightweight', ha='center', va='center', fontweight='bold')
    
    # Feature fusion
    features_box = FancyBboxPatch((0.5, 4), 2, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=input_color,
                                  edgecolor='black')
    ax.add_patch(features_box)
    ax.text(1.5, 4.5, 'ML Features\n(98d)', ha='center', va='center', fontweight='bold')
    
    fusion_box = FancyBboxPatch((3.5, 4), 3, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=fusion_color,
                                edgecolor='black')
    ax.add_patch(fusion_box)
    ax.text(5, 4.5, 'Feature Fusion\n(256d → 128d)', ha='center', va='center', fontweight='bold')
    
    # Output heads
    relevance_head = FancyBboxPatch((2, 2), 2.5, 1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=output_color,
                                    edgecolor='black')
    ax.add_patch(relevance_head)
    ax.text(3.25, 2.5, 'Relevance Head\n(4 grades)', ha='center', va='center', fontweight='bold', color='white')
    
    binary_head = FancyBboxPatch((5.5, 2), 2.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=output_color,
                                 edgecolor='black')
    ax.add_patch(binary_head)
    ax.text(6.75, 2.5, 'Binary Head\n(relevant/not)', ha='center', va='center', fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        # Input to encoders
        (1.5, 10, 1.5, 9),  # Query to Query Encoder
        (8, 10, 8, 9),      # Dataset to Dataset Encoder
        
        # Encoders to attention
        (1.5, 8, 4, 7),     # Query Encoder to Attention
        (8, 8, 6, 7),       # Dataset Encoder to Attention
        
        # Attention to fusion
        (5, 6, 5, 5),       # Attention to Fusion
        
        # Features to fusion
        (2.5, 4.5, 3.5, 4.5),  # ML Features to Fusion
        
        # Fusion to outputs
        (4.5, 4, 3.25, 3),  # Fusion to Relevance Head
        (5.5, 4, 6.75, 3),  # Fusion to Binary Head
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Title and annotations
    ax.text(5, 11.5, 'GradedRankingModel Architecture', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    
    ax.text(5, 0.5, 'Innovation: Lightweight cross-attention + hybrid scoring = 72.2% NDCG@3', 
           ha='center', va='center', fontsize=12, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig('outputs/documentation/neural_architecture_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Neural architecture diagram generated")

def create_ai_system_diagram():
    """Create AI system integration diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    user_color = '#E1F5FE'
    api_color = '#B3E5FC'
    ai_color = '#4FC3F7'
    provider_color = '#29B6F6'
    cache_color = '#0288D1'
    
    # User interface
    user_box = FancyBboxPatch((1, 8.5), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=user_color,
                              edgecolor='black')
    ax.add_patch(user_box)
    ax.text(2, 9, 'User\nInterface', ha='center', va='center', fontweight='bold')
    
    # API Gateway
    api_box = FancyBboxPatch((5, 8.5), 2, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=api_color,
                             edgecolor='black')
    ax.add_patch(api_box)
    ax.text(6, 9, 'FastAPI\nGateway', ha='center', va='center', fontweight='bold')
    
    # AI Router
    router_box = FancyBboxPatch((5, 6.5), 2, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=ai_color,
                                edgecolor='black')
    ax.add_patch(router_box)
    ax.text(6, 7, 'Query Router\n(91% accuracy)', ha='center', va='center', fontweight='bold')
    
    # AI Providers
    claude_box = FancyBboxPatch((1, 4.5), 2, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=provider_color,
                                edgecolor='black')
    ax.add_patch(claude_box)
    ax.text(2, 5, 'Claude API\n(Priority 0.9)', ha='center', va='center', fontweight='bold', color='white')
    
    mistral_box = FancyBboxPatch((5, 4.5), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=provider_color,
                                 edgecolor='black')
    ax.add_patch(mistral_box)
    ax.text(6, 5, 'Mistral API\n(Priority 0.7)', ha='center', va='center', fontweight='bold', color='white')
    
    basic_box = FancyBboxPatch((9, 4.5), 2, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=provider_color,
                               edgecolor='black')
    ax.add_patch(basic_box)
    ax.text(10, 5, 'Basic Provider\n(Always available)', ha='center', va='center', fontweight='bold', color='white')
    
    # Neural Model
    neural_box = FancyBboxPatch((1, 2.5), 3, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=ai_color,
                                edgecolor='black')
    ax.add_patch(neural_box)
    ax.text(2.5, 3, 'Neural Search Engine\n(72.2% NDCG@3)', ha='center', va='center', fontweight='bold')
    
    # Cache Layer
    cache_box = FancyBboxPatch((8, 2.5), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=cache_color,
                               edgecolor='black')
    ax.add_patch(cache_box)
    ax.text(9.5, 3, 'Cache Layer\n(66.67% hit rate)', ha='center', va='center', fontweight='bold', color='white')
    
    # Arrows showing data flow
    arrows = [
        # User to API
        (3, 9, 5, 9),
        
        # API to Router
        (6, 8.5, 6, 7.5),
        
        # Router to Providers
        (5.5, 6.5, 2.5, 5.5),  # To Claude
        (6, 6.5, 6, 5.5),      # To Mistral
        (6.5, 6.5, 9.5, 5.5),  # To Basic
        
        # Router to Neural
        (5.5, 6.5, 3, 3.5),
        
        # Cache connections
        (8, 3, 7, 4),  # Cache to providers
        
        # Return paths
        (2, 4.5, 5.5, 6.5),   # Claude back
        (6, 4.5, 6, 6.5),     # Mistral back
        (10, 4.5, 6.5, 6.5),  # Basic back
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
    
    # Title
    ax.text(6, 9.7, 'AI System Integration Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    # Performance metrics
    ax.text(6, 1, 'Performance: 84% response improvement, 99.8% fallback success, 380ms avg response',
           ha='center', va='center', fontsize=11, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    plt.tight_layout()
    plt.savefig('outputs/documentation/ai_system_diagram.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ AI system diagram generated")

if __name__ == "__main__":
    create_neural_architecture_diagram()
    create_ai_system_diagram()
