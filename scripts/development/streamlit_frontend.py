#!/usr/bin/env python3
"""
Streamlit Frontend for AI Dataset Research Assistant
Alternative frontend that bypasses JavaScript issues
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="AI Dataset Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .singapore-source {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .global-source {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🔍 AI Dataset Research Assistant")
st.markdown("**Alternative Streamlit Frontend** - Testing Singapore context prioritization")

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

def display_web_sources(web_sources: List[Dict[str, Any]]):
    """Display web sources with Singapore vs Global highlighting."""
    if not web_sources:
        st.warning("❌ No web sources returned")
        return
    
    st.subheader(f"🌐 Web Sources ({len(web_sources)})")
    
    for i, source in enumerate(web_sources, 1):
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
        
        # Create HTML container with appropriate styling
        css_class = "singapore-source" if is_singapore else "global-source"
        badge = "🇸🇬 SINGAPORE" if is_singapore else "🌍 GLOBAL"
        
        source_html = f"""
        <div class="{css_class}">
            <strong>{i}. {title}</strong><br>
            <small>Type: {source_type} | Score: {score:.1f} | {badge}</small><br>
            <a href="{url}" target="_blank">{url}</a>
        </div>
        """
        
        st.markdown(source_html, unsafe_allow_html=True)

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display dataset recommendations."""
    if not recommendations:
        return
    
    st.subheader(f"📊 Dataset Recommendations ({len(recommendations)})")
    
    for i, rec in enumerate(recommendations[:5], 1):
        dataset = rec.get('dataset', {})
        title = dataset.get('title', 'Untitled')
        source = dataset.get('source', 'Unknown')
        confidence = rec.get('confidence', 0)
        url = dataset.get('url', '#')
        
        with st.expander(f"{i}. {title}"):
            st.write(f"**Source:** {source}")
            st.write(f"**Confidence:** {confidence:.1%}")
            if url != '#':
                st.write(f"**URL:** [Open Dataset]({url})")
            
            description = dataset.get('description', '')
            if description:
                st.write(f"**Description:** {description[:200]}...")

# Search interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'housing singapore', 'transport', 'HDB', 'climate change'",
        key="search_input"
    )

with col2:
    search_button = st.button("🔍 Search", type="primary")

# Quick query buttons
st.markdown("**Quick Tests:**")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("singapore"):
        st.session_state.search_input = "singapore"
        query = "singapore"
        search_button = True

with col2:
    if st.button("housing"):
        st.session_state.search_input = "housing"
        query = "housing"
        search_button = True

with col3:
    if st.button("HDB"):
        st.session_state.search_input = "HDB"
        query = "HDB"
        search_button = True

with col4:
    if st.button("transport"):
        st.session_state.search_input = "transport"
        query = "transport"
        search_button = True

with col5:
    if st.button("climate change"):
        st.session_state.search_input = "climate change"
        query = "climate change"
        search_button = True

# Perform search
if search_button and query:
    with st.spinner(f"🔄 Searching for '{query}'..."):
        start_time = time.time()
        data = call_api(query)
        search_time = time.time() - start_time
    
    if data:
        # Display metrics
        st.markdown(f"""
        <div class="metric-container">
            <strong>Search Results for:</strong> "{query}"<br>
            <strong>Processing Time:</strong> {search_time:.2f}s<br>
            <strong>Web Sources:</strong> {len(data.get('web_sources', []))}<br>
            <strong>Dataset Recommendations:</strong> {len(data.get('recommendations', []))}<br>
            <strong>AI Provider:</strong> {data.get('performance', {}).get('ai_provider', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_web_sources(data.get('web_sources', []))
        
        with col2:
            display_recommendations(data.get('recommendations', []))
        
        # Show raw JSON for debugging (collapsible)
        with st.expander("🔧 Raw API Response (Debug)"):
            st.json(data)

# Instructions
st.sidebar.markdown("""
## 🎯 Testing Instructions

1. **Test Singapore Context:**
   - Try: "singapore", "housing", "HDB"
   - **Expected:** Green Singapore sources first

2. **Test Global Context:**
   - Try: "climate change"
   - **Expected:** Yellow global sources first

3. **Check Results:**
   - Singapore sources should be highlighted in green
   - Global sources should be highlighted in yellow
   - Scores should show Singapore > 5000, Global < 1000

## 🐛 Debugging

If you see issues:
- Check processing time (should be < 10s)
- Check AI provider (should be "claude")
- Check web sources count (should be > 0)
- Green sources should appear first for Singapore queries
""")

# Footer
st.markdown("---")
st.markdown("**Note:** This Streamlit frontend bypasses all JavaScript caching and routing issues.")