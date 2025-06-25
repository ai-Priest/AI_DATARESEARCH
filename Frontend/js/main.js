// Singapore Dataset Discovery Assistant - Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';
const chatArea = document.getElementById('chatArea');
const searchInput = document.getElementById('searchInput');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners
    setupEventListeners();
    
    // Display welcome message
    displayWelcomeMessage();
});

function setupEventListeners() {
    // Search input enter key
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Example query clicks
    document.querySelectorAll('.example-queries li').forEach(item => {
        item.addEventListener('click', function() {
            const query = this.textContent.replace(/['"]/g, '');
            searchInput.value = query;
            performSearch();
        });
    });
}

function displayWelcomeMessage() {
    if (!chatArea) return;
    
    chatArea.innerHTML = `
        <div class="chat-message">
            <strong>AI Assistant:</strong> Welcome! I'm ready to help you find Singapore datasets. 
            Try searching for topics like "transport", "housing", "education", or ask specific questions.
        </div>
    `;
}

async function performSearch(query = null) {
    const searchQuery = query || searchInput.value.trim();
    
    if (!searchQuery) {
        alert('Please enter a search query');
        return;
    }
    
    // Clear previous results and show loading
    displayLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/ai-search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: searchQuery,
                use_ai_enhanced_search: true,
                top_k: 8
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displaySearchResults(data, searchQuery);
        
    } catch (error) {
        console.error('Search error:', error);
        displayError('Sorry, there was an error processing your search. Please try again.');
    }
}

function displayLoading() {
    if (!chatArea) return;
    
    chatArea.innerHTML = `
        <div class="chat-message">
            <strong>AI Assistant:</strong> <span class="loading">Searching datasets</span>
        </div>
    `;
}

function displayError(message) {
    if (!chatArea) return;
    
    chatArea.innerHTML = `
        <div class="chat-message" style="border-left-color: #ef4444;">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

function displaySearchResults(data, query) {
    if (!chatArea) return;
    
    const recommendations = data.recommendations || [];
    const webSources = data.web_sources || [];
    const response = data.response || `Found ${recommendations.length} datasets`;
    
    let html = `
        <div class="chat-message">
            <strong>AI Assistant:</strong> ${response}
        </div>
    `;
    
    if (recommendations.length > 0) {
        html += `
            <div class="dataset-recommendations">
                <div style="border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin: 15px 0; background: #f8fafc;">
                    <h3 style="color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        🎯 <span>Dataset Recommendations</span>
                    </h3>
                    ${recommendations.map((rec, index) => createDatasetCard(rec, index)).join('')}
                </div>
            </div>
        `;
    }
    
    if (webSources.length > 0) {
        html += `
            <div class="web-sources">
                <div style="border: 2px solid #10b981; border-radius: 12px; padding: 20px; margin: 15px 0; background: #f0fdf4;">
                    <h3 style="color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        🌐 <span>Additional Web Sources</span>
                    </h3>
                    ${webSources.map((source, index) => createWebSourceCard(source, index)).join('')}
                </div>
            </div>
        `;
    }
    
    chatArea.innerHTML = html;
    chatArea.scrollTop = chatArea.scrollHeight;
}

function createDatasetCard(recommendation, index) {
    // Handle both direct result and recommendation structure
    const dataset = recommendation.dataset || recommendation;
    const confidence = recommendation.confidence || 0.5;
    
    const title = dataset.title || dataset.name || 'Untitled Dataset';
    const description = dataset.description || 'No description available';
    const agency = dataset.agency || dataset.source || 'Unknown Agency';
    const format = dataset.format || 'Unknown';
    const lastUpdated = dataset.last_updated || 'Unknown';
    
    let url = dataset.url || '#';
    
    // Ensure URL is complete with protocol
    if (url !== '#' && !url.startsWith('http')) {
        url = 'https://' + url;
    }
    
    const relevancePercentage = Math.round(confidence * 100);
    
    return `
        <div class="dataset-card" style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <h4 style="color: #1e293b; margin: 0; flex: 1;">${index + 1}. ${title}</h4>
                <span class="relevance-badge">${relevancePercentage}%</span>
            </div>
            
            <p style="color: #64748b; margin-bottom: 10px; font-size: 0.9rem; line-height: 1.4;">
                ${description.length > 200 ? description.substring(0, 200) + '...' : description}
            </p>
            
            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 12px; font-size: 0.8rem; color: #64748b;">
                <span>🏢 ${agency}</span>
                <span>📄 ${format}</span>
                <span>📅 ${lastUpdated}</span>
            </div>
            
            <div style="text-align: center;">
                <a href="${url}" target="_blank" class="view-dataset-btn" 
                   style="background: #3b82f6; color: white; text-decoration: none; padding: 10px 20px; 
                          border-radius: 6px; font-weight: 500; display: inline-block; transition: all 0.3s ease;"
                   onmouseover="this.style.background='#2563eb'" 
                   onmouseout="this.style.background='#3b82f6'">
                    View Dataset 🔗
                </a>
            </div>
        </div>
    `;
}

// Global function for quick action buttons
window.performSearch = performSearch;

// Add some utility functions
function formatDate(dateString) {
    if (!dateString || dateString === 'Unknown') return 'Unknown';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-SG', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return dateString;
    }
}

function createWebSourceCard(source, index) {
    const title = source.title || 'Untitled Source';
    const url = source.url || '#';
    const description = source.description || 'No description available';
    const sourceType = source.source || 'web';
    const type = source.type || 'general';
    const relevanceScore = source.relevance_score || 0;
    
    // Truncate description if too long
    const shortDescription = description.length > 150 ? 
        description.substring(0, 150) + '...' : description;
    
    // Get source type icon
    const getSourceIcon = (sourceType, type) => {
        if (sourceType === 'government_portal' || type === 'government_data') return '🏛️';
        if (sourceType === 'zenodo' || type === 'academic_search') return '🎓';
        if (sourceType === 'figshare') return '📊';
        if (sourceType === 'duckduckgo') return '🔍';
        return '🌐';
    };
    
    const icon = getSourceIcon(sourceType, type);
    
    return `
        <div style="
            background: white; 
            border: 1px solid #d1d5db; 
            border-radius: 8px; 
            padding: 16px; 
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        " onmouseover="this.style.borderColor='#10b981'; this.style.boxShadow='0 4px 12px rgba(16,185,129,0.15)'" 
           onmouseout="this.style.borderColor='#d1d5db'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'">
            
            <div style="margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">${icon}</span>
                    <h4 style="margin: 0; color: #1e293b; font-size: 1.1em; font-weight: 600;">
                        ${sanitizeHtml(title)}
                    </h4>
                    ${relevanceScore > 0 ? `
                        <span style="
                            background: #10b981; 
                            color: white; 
                            padding: 2px 8px; 
                            border-radius: 12px; 
                            font-size: 0.8em; 
                            font-weight: 500;
                            margin-left: auto;
                        ">
                            ${Math.round(relevanceScore)}% relevant
                        </span>
                    ` : ''}
                </div>
                
                <p style="margin: 8px 0; color: #64748b; font-size: 0.95em; line-height: 1.4;">
                    ${sanitizeHtml(shortDescription)}
                </p>
                
                <div style="display: flex; align-items: center; gap: 12px; margin-top: 12px;">
                    <span style="
                        background: #f1f5f9; 
                        color: #475569; 
                        padding: 4px 8px; 
                        border-radius: 6px; 
                        font-size: 0.8em; 
                        font-weight: 500;
                    ">
                        ${sourceType.replace('_', ' ').toUpperCase()}
                    </span>
                    
                    <a href="${url}" target="_blank" style="
                        color: #10b981; 
                        text-decoration: none; 
                        font-weight: 500; 
                        font-size: 0.9em;
                        border: 1px solid #10b981;
                        padding: 6px 12px;
                        border-radius: 6px;
                        transition: all 0.2s ease;
                        margin-left: auto;
                    " onmouseover="this.style.background='#10b981'; this.style.color='white'" 
                       onmouseout="this.style.background='transparent'; this.style.color='#10b981'">
                        Visit Source →
                    </a>
                </div>
            </div>
        </div>
    `;
}

function sanitizeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Health check function
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            console.log('✅ Backend server is healthy');
            return true;
        }
    } catch (error) {
        console.warn('⚠️ Backend server may be offline:', error.message);
        return false;
    }
    return false;
}

// Check server health on load
setTimeout(checkServerHealth, 1000);