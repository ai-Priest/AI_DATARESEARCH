#!/usr/bin/env python3
"""
Improved Streamlit Frontend for AI Dataset Research Assistant
Matches original frontend design with better functionality
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="AI Dataset Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match original design
st.markdown("""
<style>
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Search container */
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    /* Result card styling */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .singapore-card {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #f8fff9 0%, #e8f5e8 100%);
    }
    
    .global-card {
        border-left: 5px solid #007bff;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .result-meta {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .result-url {
        color: #007bff;
        text-decoration: none;
        font-size: 0.9rem;
        word-break: break-all;
    }
    
    .singapore-badge {
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 10px;
    }
    
    .global-badge {
        background: #007bff;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 10px;
    }
    
    .confidence-badge {
        background: #6c757d;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    /* Stats container */
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    /* Quick buttons */
    .quick-button {
        background: #e9ecef;
        border: none;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .quick-button:hover {
        background: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
<div class="header-container">
    <div class="header-title">📊 AI Dataset Research Assistant</div>
    <div class="header-subtitle">Intelligent dataset discovery with Singapore context prioritization</div>
</div>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def call_api(query: str) -> Dict[str, Any]:
    """Call the AI search API with the given query."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ai-search",
            json={
                "query": query,
                "use_ai_enhanced_search": True,
                "top_k": 8
            },
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Make sure it's running on localhost:8000")
        return {}
    except requests.exceptions.Timeout:
        st.error("⏰ API request timed out. Try again.")
        return {}
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return {}

def is_query_relevant_to_dataset(query: str, dataset: Dict[str, Any]) -> bool:
    """Check if a dataset is relevant to the query."""
    query_lower = query.lower()
    dataset_title = dataset.get('title', '').lower()
    dataset_desc = dataset.get('description', '').lower()
    
    # Extract key terms from query
    query_terms = set(query_lower.split())
    common_words = {'data', 'dataset', 'information', 'statistics', 'research', 'analysis', 'the', 'and', 'or', 'for', 'in', 'on', 'at', 'to', 'from'}
    relevant_query_terms = query_terms - common_words
    
    # Check if any relevant query terms appear in dataset title or description
    for term in relevant_query_terms:
        if term in dataset_title or term in dataset_desc:
            return True
    
    return False

def display_combined_results(data: Dict[str, Any], query: str):
    """Display combined web sources and relevant dataset recommendations."""
    web_sources = data.get('web_sources', [])
    recommendations = data.get('recommendations', [])
    
    # Filter relevant recommendations
    relevant_recommendations = []
    for rec in recommendations:
        dataset = rec.get('dataset', {})
        if is_query_relevant_to_dataset(query, dataset):
            relevant_recommendations.append(rec)
    
    # Combine and sort by relevance/confidence
    combined_results = []
    
    # Add web sources
    for source in web_sources:
        combined_results.append({
            'type': 'web_source',
            'data': source,
            'score': source.get('relevance_score', 0)
        })
    
    # Add relevant dataset recommendations
    for rec in relevant_recommendations:
        combined_results.append({
            'type': 'dataset',
            'data': rec,
            'score': rec.get('confidence', 0) * 1000  # Scale confidence to match relevance scores
        })
    
    # Sort by score (highest first)
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    if not combined_results:
        st.warning("❌ No relevant results found for your query")
        return
    
    # Display stats
    web_count = len(web_sources)
    dataset_count = len(relevant_recommendations)
    total_count = len(combined_results)
    
    st.markdown(f"""
    <div class="stats-container">
        <strong>Search Results for:</strong> "{query}"<br>
        <strong>Total Sources:</strong> {total_count} ({web_count} web sources + {dataset_count} relevant datasets)<br>
        <strong>Processing Time:</strong> {data.get('processing_time', 0):.2f}s<br>
        <strong>AI Provider:</strong> {data.get('performance', {}).get('ai_provider', 'Unknown')}
    </div>
    """, unsafe_allow_html=True)
    
    # Display combined results
    st.subheader(f"📊 All Results ({total_count})")
    
    for i, result in enumerate(combined_results, 1):
        result_type = result['type']
        result_data = result['data']
        
        if result_type == 'web_source':
            display_web_source_card(result_data, i)
        else:  # dataset
            display_dataset_card(result_data, i)

def display_web_source_card(source: Dict[str, Any], index: int):
    """Display a web source as a card."""
    title = source.get('title', 'Untitled')
    url = source.get('url', '#')
    source_type = source.get('type', 'Unknown')
    score = source.get('relevance_score', 0)
    
    # Determine if this is a Singapore source
    is_singapore = (
        source_type == 'government_data' or
        'data.gov.sg' in url or
        'singstat.gov.sg' in url or
        'singapore' in title.lower()
    )
    
    card_class = "singapore-card" if is_singapore else "global-card"
    badge = "🇸🇬 SINGAPORE" if is_singapore else "🌍 GLOBAL"
    badge_class = "singapore-badge" if is_singapore else "global-badge"
    
    # Check URL status
    url_status = "🔗" if url and url != '#' else "❌"
    
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-title">{index}. {title}</div>
        <div class="result-meta">
            <span class="{badge_class}">{badge}</span>
            <span class="confidence-badge">Score: {score:.0f}</span>
        </div>
        <div class="result-meta">Type: {source_type} | Status: {url_status}</div>
        <a href="{url}" target="_blank" class="result-url">{url}</a>
    </div>
    """, unsafe_allow_html=True)

def display_dataset_card(rec: Dict[str, Any], index: int):
    """Display a dataset recommendation as a card."""
    dataset = rec.get('dataset', {})
    title = dataset.get('title', 'Untitled')
    source = dataset.get('source', 'Unknown')
    confidence = rec.get('confidence', 0)
    url = dataset.get('url', '#')
    description = dataset.get('description', '')
    
    # Determine if this is a Singapore dataset
    is_singapore = (
        'data.gov.sg' in source.lower() or
        'singstat' in source.lower() or
        'hdb' in source.lower() or
        'singapore' in title.lower()
    )
    
    card_class = "singapore-card" if is_singapore else "global-card"
    badge = "🇸🇬 SINGAPORE DATASET" if is_singapore else "📊 DATASET"
    badge_class = "singapore-badge" if is_singapore else "global-badge"
    
    # Truncate description
    short_desc = description[:150] + "..." if len(description) > 150 else description
    
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-title">{index}. {title}</div>
        <div class="result-meta">
            <span class="{badge_class}">{badge}</span>
            <span class="confidence-badge">Confidence: {confidence:.1%}</span>
        </div>
        <div class="result-meta">Source: {source}</div>
        <div class="result-meta">{short_desc}</div>
        <a href="{url}" target="_blank" class="result-url">{url}</a>
    </div>
    """, unsafe_allow_html=True)

# Search interface
st.markdown('<div class="search-container">', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Search for datasets:",
        placeholder="e.g., 'housing singapore', 'laptop prices', 'climate data', 'transport'",
        key="search_input",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Quick query buttons
st.markdown("**Quick Tests:**")
cols = st.columns(6)
quick_queries = ["singapore", "housing", "HDB", "laptop prices", "health", "climate change"]

for i, q in enumerate(quick_queries):
    with cols[i]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.search_input = q
            query = q
            search_button = True

st.markdown('</div>', unsafe_allow_html=True)

# Perform search
if search_button and query:
    with st.spinner(f"🔄 Searching for '{query}'..."):
        start_time = time.time()
        data = call_api(query)
        search_time = time.time() - start_time
    
    if data:
        display_combined_results(data, query)
        
        # Show debug info in expander
        with st.expander("🔧 Debug Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "query": query,
                    "web_sources_count": len(data.get('web_sources', [])),
                    "recommendations_count": len(data.get('recommendations', [])),
                    "processing_time": f"{search_time:.2f}s",
                    "ai_provider": data.get('performance', {}).get('ai_provider', 'Unknown')
                })
            with col2:
                st.json({
                    "first_web_source": data.get('web_sources', [{}])[0].get('title', 'None') if data.get('web_sources') else 'None',
                    "first_recommendation": data.get('recommendations', [{}])[0].get('dataset', {}).get('title', 'None') if data.get('recommendations') else 'None',
                    "singapore_context_detected": any(kw in query.lower() for kw in ['singapore', 'sg', 'hdb', 'housing'])
                })

# Footer
st.markdown("---")
st.markdown("**✨ Enhanced Streamlit Frontend** - Combines web sources and relevant datasets with improved styling")