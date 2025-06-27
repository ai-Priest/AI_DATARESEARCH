// Singapore Dataset Discovery Assistant - Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';
const chatArea = document.getElementById('chatArea');
const resultsArea = document.getElementById('resultsArea');
const resultsCount = document.getElementById('resultsCount');
const resultsPanel = document.getElementById('resultsPanel');
const searchInput = document.getElementById('searchInput');
const mainContent = document.querySelector('.main-content');

// Disable search history to force fresh searches
// let searchHistory = [];
// const MAX_HISTORY_SIZE = 10;

// Conversation management
let currentSessionId = null;
let conversationMode = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Skip search history loading for fresh experience
    // loadSearchHistory();
    
    // Add event listeners
    setupEventListeners();
    
    // Always display welcome message
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

// Search history disabled for fresh searches
// function loadSearchHistory() {
//     try {
//         const saved = localStorage.getItem('datasetSearchHistory');
//         if (saved) {
//             searchHistory = JSON.parse(saved);
//         }
//     } catch (error) {
//         console.log('Could not load search history:', error);
//         searchHistory = [];
//     }
// }

// Search history disabled
// function saveSearchHistory() {
//     try {
//         localStorage.setItem('datasetSearchHistory', JSON.stringify(searchHistory));
//     } catch (error) {
//         console.log('Could not save search history:', error);
//     }
// }

// History disabled - function placeholder
function addToHistory(userQuery, aiResponse, results, webSources = []) {
    // History saving disabled for fresh searches
    console.log('Search completed:', userQuery);
}

// Chat history display disabled
function displayChatHistory() {
    displayWelcomeMessage();
}

function displayWelcomeMessage() {
    if (!chatArea) return;
    
    chatArea.innerHTML = `
        <div class="welcome-message">
            <div class="ai-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-text">
                        How can I help you today?
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // No history tracking
}

function addClearChatButton() {
    if (!chatArea) return;
    
    const clearButton = document.createElement('div');
    clearButton.innerHTML = `
        <div style="text-align: center; margin: 20px 0;">
            <button onclick="clearChatHistory()" 
                    style="background: #ef4444; color: white; border: none; padding: 8px 16px; 
                           border-radius: 6px; font-size: 0.85rem; cursor: pointer; transition: background 0.2s;"
                    onmouseover="this.style.background='#dc2626'" 
                    onmouseout="this.style.background='#ef4444'"
                    title="Clear all chat history">
                🗑️ Clear Chat History
            </button>
        </div>
    `;
    chatArea.appendChild(clearButton);
}

function clearChatHistory() {
    // Clear current display and reset
    displayWelcomeMessage();
    
    // Close results panel if open
    if (resultsPanel && resultsPanel.classList.contains('visible')) {
        closeResultsPanel();
    }
    
    // Reset session
    currentSessionId = null;
    conversationMode = false;
}

function isConversationalQuery(query) {
    const conversationalKeywords = [
        'hello', 'how are you', 'what can you do', 'help me understand',
        'thanks', 'thank you', 'good morning', 'good afternoon', 'goodbye'
    ];
    
    const lowerQuery = query.toLowerCase();
    
    // Check for explicit conversational keywords (exact match or at word boundaries)
    if (conversationalKeywords.some(keyword => {
        return lowerQuery === keyword || 
               lowerQuery.startsWith(keyword + ' ') || 
               lowerQuery.endsWith(' ' + keyword) ||
               lowerQuery.includes(' ' + keyword + ' ');
    })) {
        return true;
    }
    
    // Also check for standalone "hi" (but not as part of other words like HIV)
    if (lowerQuery === 'hi' || lowerQuery === 'hi there' || lowerQuery === 'hey') {
        return true;
    }
    
    // Check for non-data related phrases that should be conversational
    const nonDataPhrases = [
        'money please', 'give me money', 'i need money', 'pay me', 'send money',
        'i love you', 'you are cute', 'marry me', 'be my friend',
        'what\'s up', 'how\'s it going', 'good job', 'well done',
        'i hate', 'i don\'t like', 'this sucks', 'boring',
        'lol', 'haha', 'funny', 'joke', 'kidding'
    ];
    
    if (nonDataPhrases.some(phrase => lowerQuery.includes(phrase))) {
        return true;
    }
    
    // Check if query is too short or looks like random text (only for very short queries)
    const words = lowerQuery.trim().split(/\s+/);
    // Don't treat legitimate single-word searches as conversation
    // Only treat as conversation if it's a greeting or clearly non-data query
    if (words.length <= 1) {
        const greetings = ['hi', 'hello', 'hey', 'bye', 'thanks', 'ok', 'yes', 'no'];
        if (greetings.includes(words[0])) {
            return true;
        }
    }
    
    // Default to search behavior - let the AI search handle most queries
    return false;
}

async function performSearch(query = null) {
    const searchQuery = query || searchInput.value.trim();
    
    if (!searchQuery) {
        alert('Please enter a search query');
        return;
    }
    
    // Determine if this should be a conversation or search
    const shouldConverse = isConversationalQuery(searchQuery);
    
    if (shouldConverse) {
        await handleConversation(searchQuery);
    } else {
        await handleDatasetSearch(searchQuery);
    }
}

async function handleConversation(message) {
    displayLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/conversation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId,
                include_search: message.toLowerCase().includes('dataset') || message.toLowerCase().includes('data')
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayConversationResult(data, message);
        
    } catch (error) {
        console.error('Conversation error:', error);
        displayError('Sorry, I had trouble understanding you. Please try again.');
    }
}

async function handleDatasetSearch(query) {
    displayLoading();
    
    try {
        // Try AI search first, fallback to simple search if it fails
        let response;
        let data;
        let useAISearch = true;
        
        try {
            response = await fetch(`${API_BASE_URL}/api/ai-search?t=${Date.now()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                },
                body: JSON.stringify({
                    query: query,
                    use_ai_enhanced_search: true,
                    top_k: 8
                })
            });
            
            if (!response.ok) throw new Error('AI search failed');
            data = await response.json();
            
            // Debug logging for troubleshooting
            console.log('🔍 AI Search Response:', {
                web_sources_count: data.web_sources?.length || 0,
                first_web_source: data.web_sources?.[0]?.title || 'None',
                first_web_source_type: data.web_sources?.[0]?.type || 'None'
            });
            
            // Don't throw error if no local recommendations - web sources are still valid
            // Remove this check entirely as we now use unified results
            
        } catch (aiError) {
            console.log('AI search failed, falling back to simple search:', aiError.message);
            useAISearch = false;
            
            // Fallback to simple search
            response = await fetch(`${API_BASE_URL}/api/search?t=${Date.now()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                },
                body: JSON.stringify({
                    query: query,
                    top_k: 8
                })
            });
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Get data if not already fetched from AI search
        if (!data) {
            data = await response.json();
        }
        
        console.log('Search response:', data); // Debug logging
        console.log('Using AI search:', useAISearch);
        console.log('Has all_results:', data.all_results ? data.all_results.length : 'none');
        
        // Handle both AI search and simple search responses
        if (useAISearch) {
            // AI search response format: {recommendations: [...], web_sources: [...], response: "..."}
            displaySearchResults(data, query);
        } else {
            // Simple search response format: {results: [...]}
            // Convert to AI search format for consistent display
            const aiFormattedData = {
                recommendations: data.results.map(result => ({
                    dataset: result,
                    confidence: result.relevance_score || 0.7,
                    explanation: `${result.title} matches your query with high relevance.`
                })),
                web_sources: [
                    // Add HDB official source for housing queries
                    ...(query.toLowerCase().includes('hdb') || query.toLowerCase().includes('housing') ? [{
                        title: `HDB Official Website - ${query} Information`,
                        url: `https://www.hdb.gov.sg/`,
                        description: `Official HDB resources and information about ${query}`,
                        source: 'hdb_official',
                        type: 'singapore_government',
                        relevance_score: 95
                    }] : []),
                    {
                        title: `Singapore Open Data - ${query} Datasets`,
                        url: `https://data.gov.sg/datasets?agencies=Housing+%26+Development+Board+(HDB)&resultId=189&page=1`,
                        description: `Government datasets related to ${query} from Singapore Open Data portal`,
                        source: 'singapore_government',
                        type: 'singapore_government', 
                        relevance_score: 90
                    },
                    {
                        title: `Search World Bank for ${query} data`,
                        url: `https://data.worldbank.org/indicator?tab=all&q=${encodeURIComponent(query)}`,
                        description: `Global datasets and indicators from World Bank Open Data`,
                        source: 'world_bank',
                        type: 'global_data',
                        relevance_score: 85
                    }
                ],
                response: `Great! I found ${data.results.length} relevant datasets and 3 web sources for your search. Check the results panel on the right →`,
                session_id: `search-${Date.now()}`,
                query: query
            };
            displaySearchResults(aiFormattedData, query);
        }
        
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

function displayConversationResult(data, userMessage) {
    if (!chatArea) return;
    
    // Store session ID for future conversations
    if (data.session_id) {
        currentSessionId = data.session_id;
    }
    
    // Generate AI response
    const aiResponse = data.response || "I'm here to help you with Singapore datasets!";
    
    // Append conversation to existing chat
    const conversationHtml = `
        <div class="user-message">
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <div class="message-text">${userMessage}</div>
                <div class="message-timestamp">${formatTimestamp(new Date().toISOString())}</div>
            </div>
        </div>
        <div class="ai-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">${aiResponse}</div>
            </div>
        </div>
    `;
    
    chatArea.innerHTML += conversationHtml;
    chatArea.scrollTop = chatArea.scrollHeight;
    
    // If there are search results from the conversation, show them
    if (data.search_results && data.search_results.length > 0) {
        showSearchResultsFromConversation(data.search_results);
    }
    
    // Add to search history
    addToHistory(userMessage, aiResponse, data.search_results || [], []);
    
    // Clear search input for next message
    searchInput.value = '';
}

function showSearchResultsFromConversation(searchResults) {
    if (!resultsArea || !resultsPanel) return;
    
    // Show results panel
    resultsPanel.style.display = 'block';
    setTimeout(() => {
        resultsPanel.classList.add('visible');
        if (mainContent) {
            mainContent.classList.add('with-results');
        }
        const hero = document.querySelector('.hero');
        if (hero) {
            hero.classList.add('with-results');
        }
    }, 10);
    
    // Update results count
    resultsCount.textContent = `${searchResults.length} datasets found`;
    
    // Display search results as dataset cards
    const resultsHtml = `
        <div class="datasets-section">
            <h4 style="color: #3b82f6; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
                📊 <span>Related Datasets</span>
            </h4>
            ${searchResults.map((result, index) => createDatasetCard(result, index)).join('')}
        </div>
    `;
    
    resultsArea.innerHTML = resultsHtml;
    resultsArea.scrollTop = 0;
}

function displaySearchResults(data, query) {
    if (!chatArea || !resultsArea || !resultsPanel) return;
    
    // Handle both AI search and simple search response formats
    const recommendations = data.recommendations || data.results || [];
    // Use actual web sources from API, only fallback if missing or empty
    let webSources = data.web_sources;
    
    // Only use fallback if no web sources returned from API
    if (!webSources || webSources.length === 0) {
        webSources = [
        // Add Singapore-specific web sources for HDB queries
        ...(query.toLowerCase().includes('hdb') || query.toLowerCase().includes('housing') ? [
            {
                title: `HDB Official Website - ${query} Information`,
                url: `https://www.hdb.gov.sg/`,
                description: `Official HDB resources and information about ${query}`,
                source: 'hdb_official',
                type: 'singapore_government',
                relevance_score: 95
            },
            {
                title: `Singapore Open Data - ${query} Datasets`,
                url: `https://data.gov.sg/search?q=${encodeURIComponent(query)}`,
                description: `Government datasets related to ${query} from Singapore Open Data portal`,
                source: 'singapore_government',
                type: 'singapore_government', 
                relevance_score: 90
            }
        ] : []),
        // Add global sources
        {
            title: `Search World Bank for ${query} data`,
            url: `https://data.worldbank.org/search?q=${encodeURIComponent(query)}`,
            description: `Global datasets and indicators from World Bank Open Data`,
            source: 'world_bank',
            type: 'global_data',
            relevance_score: 85
        }
        ];
    }
    
    // Generate AI response
    const datasetCount = recommendations?.length || 0;
    const webSourceCount = webSources?.length || 0;
    let aiResponse = '';
    
    if (datasetCount > 0 && webSourceCount > 0) {
        aiResponse = `Great! I found <strong>${datasetCount} relevant datasets</strong> and <strong>${webSourceCount} web sources</strong> for your search. Check the results panel on the right →`;
    } else if (datasetCount > 0) {
        aiResponse = `I found <strong>${datasetCount} relevant datasets</strong> for "${query}". Check the results panel on the right →`;
    } else if (webSourceCount > 0) {
        aiResponse = `I found <strong>${webSourceCount} web sources</strong> for your search. Check the results panel on the right →`;
    } else {
        aiResponse = `I couldn't find any datasets matching "${query}". Try using different search terms or check the suggestions.`;
    }
    
    // Append new conversation to existing chat
    const newChatHtml = `
        <div class="user-message">
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <div class="message-text">${query}</div>
                <div class="message-timestamp">${formatTimestamp(new Date().toISOString())}</div>
            </div>
        </div>
        <div class="ai-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">${aiResponse}</div>
            </div>
        </div>
    `;
    
    chatArea.innerHTML += newChatHtml;
    chatArea.scrollTop = chatArea.scrollHeight;
    
    // Add to search history
    addToHistory(query, aiResponse.replace(/<[^>]*>/g, ''), recommendations, webSources);
    
    // Clear search input for next query
    searchInput.value = '';
    
    // Show results panel with sliding animation and adjust main content
    resultsPanel.style.display = 'block';
    setTimeout(() => {
        resultsPanel.classList.add('visible');
        if (mainContent) {
            mainContent.classList.add('with-results');
        }
        // Also adjust header
        const hero = document.querySelector('.hero');
        if (hero) {
            hero.classList.add('with-results');
        }
    }, 10);
    
    // Update results count - show both datasets and web sources
    const totalCount = datasetCount + webSourceCount;
    resultsCount.textContent = totalCount > 0 
        ? `${totalCount} sources found`
        : 'No sources found';
    
    // Display unified results (if API provides all_results, use that; otherwise fall back to old format)
    let resultsHtml = '';
    const allResults = data.all_results;
    
    console.log('Displaying results - allResults:', allResults); // Debug
    console.log('Recommendations count:', recommendations?.length || 0);
    console.log('Web sources count:', webSources?.length || 0);
    
    if (allResults && allResults.length > 0) {
        console.log('Using unified results display');
        // Use new unified results - no separate sections
        resultsHtml += `
            <div class="unified-results-section">
                ${allResults.map((result, index) => {
                    // Determine which card type to use based on result type
                    if (result.type === 'local_dataset' && result.dataset_info) {
                        // Local dataset with full info
                        return createDatasetCard({
                            dataset: result.dataset_info,
                            confidence: result.confidence,
                            explanation: result.explanation,
                            source: result.source
                        }, index);
                    } else {
                        // Web source or simple result
                        return createWebSourceCard(result, index);
                    }
                }).join('')}
            </div>
        `;
    } else {
        // Fallback to old format for backward compatibility
        // Show dataset recommendations first (if any)
        if (recommendations && recommendations.length > 0) {
            resultsHtml += `
                <div class="datasets-section">
                    ${recommendations.map((rec, index) => createDatasetCard(rec, index)).join('')}
                </div>
            `;
        }
        
        // Then show web sources (if any)
        if (webSources && webSources.length > 0) {
            if (resultsHtml) resultsHtml += '<div style="margin: 30px 0;"></div>'; // Add spacing between sections
            resultsHtml += `
                <div class="web-sources-section">
                    ${webSources.map((source, index) => createWebSourceCard(source, index)).join('')}
                </div>
            `;
        }
    }
    
    // If no results at all
    if (!resultsHtml || resultsHtml.trim() === '') {
        console.log('No results HTML generated, showing no results message');
        // Show helpful suggestions when no results found
        resultsHtml = `
            <div class="no-results-panel">
                <div class="no-results-content">
                    <h3 style="color: #f59e0b; margin-bottom: 15px;">🔍 No datasets found for "${query}"</h3>
                    <p style="color: #64748b; margin-bottom: 20px;">
                        Try these suggestions to find what you're looking for:
                    </p>
                    
                    <div class="search-tips">
                        <h4 style="color: #1e293b; margin-bottom: 10px;">💡 Search Tips:</h4>
                        <ul style="color: #64748b; margin-left: 20px; margin-bottom: 20px;">
                            <li>Use broader terms (e.g., "housing" instead of "HDB resale prices")</li>
                            <li>Check spelling and try synonyms</li>
                            <li>Use Singapore-specific terms (e.g., "MRT", "HDB", "CPF")</li>
                        </ul>
                    </div>
                    
                    <div class="popular-topics">
                        <h4 style="color: #1e293b; margin-bottom: 10px;">🏷️ Popular Topics:</h4>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            ${['transport', 'housing', 'education', 'healthcare', 'economy', 'population', 'environment', 'tourism'].map(topic => 
                                `<button onclick="performSearch('${topic}')" style="background: #f59e0b; color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; transition: background 0.3s;" onmouseover="this.style.background='#d97706'" onmouseout="this.style.background='#f59e0b'">${topic}</button>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    resultsArea.innerHTML = resultsHtml;
    resultsArea.scrollTop = 0;
}

function createDatasetCard(recommendation, index) {
    // Handle both direct result and recommendation structure
    const dataset = recommendation.dataset || recommendation;
    const confidence = dataset.confidence || recommendation.confidence || 0.5;
    
    const title = dataset.title || dataset.name || 'Untitled Dataset';
    const description = dataset.description || 'No description available';
    const agency = dataset.agency || dataset.source || 'Government Source';
    const format = dataset.format && dataset.format !== 'Unknown' ? dataset.format : null;
    const lastUpdated = dataset.last_updated && dataset.last_updated !== 'Unknown' ? formatDate(dataset.last_updated) : null;
    
    let url = dataset.url || '#';
    
    // Ensure URL is complete with protocol
    if (url !== '#' && !url.startsWith('http')) {
        url = 'https://' + url;
    }
    
    // Cap relevance percentage at 100% to avoid confusion
    const relevancePercentage = Math.min(Math.round(confidence * 100), 100);
    
    // Generate explanation for why this dataset is recommended
    const explanation = generateRecommendationExplanation(recommendation, dataset, confidence);
    
    return `
        <div class="dataset-card" style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <h4 style="color: #1e293b; margin: 0; flex: 1; font-size: 1.1rem;">${index + 1}. ${title}</h4>
                <span style="background: #3b82f6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 500;">${relevancePercentage}%</span>
            </div>
            
            <p style="color: #64748b; margin-bottom: 15px; font-size: 0.9rem; line-height: 1.5;">
                ${description.length > 150 ? description.substring(0, 150) + '...' : description}
            </p>
            
            <!-- Recommendation explanation -->
            <div style="background: #f8fafc; border-radius: 8px; padding: 12px; margin-bottom: 15px; border-left: 3px solid #3b82f6;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 500; margin-bottom: 4px;">💡 Why this matches:</div>
                <div style="font-size: 0.8rem; color: #64748b; line-height: 1.3;">${explanation}</div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 15px; font-size: 0.75rem; color: #64748b;">
                <span style="background: #f1f5f9; padding: 4px 8px; border-radius: 6px;">🏢 ${agency}</span>
                ${format ? `<span style="background: #f1f5f9; padding: 4px 8px; border-radius: 6px;">📄 ${format}</span>` : ''}
                ${lastUpdated ? `<span style="background: #f1f5f9; padding: 4px 8px; border-radius: 6px;">📅 ${lastUpdated}</span>` : ''}
            </div>
            
            <div style="display: flex; gap: 10px; align-items: center;">
                <a href="${url}" target="_blank" 
                   style="background: #3b82f6; color: white; text-decoration: none; padding: 12px 20px; 
                          border-radius: 8px; font-weight: 500; flex: 1; text-align: center; transition: all 0.3s ease;"
                   onmouseover="this.style.background='#2563eb'" 
                   onmouseout="this.style.background='#3b82f6'">
                    📊 View Dataset
                </a>
                <button onclick="provideFeedback('${dataset.dataset_id}', 'helpful')" 
                        style="background: #10b981; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer;" 
                        title="This was helpful">
                    👍
                </button>
                <button onclick="provideFeedback('${dataset.dataset_id}', 'not_helpful')" 
                        style="background: #ef4444; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer;" 
                        title="This wasn't helpful">
                    👎
                </button>
            </div>
        </div>
    `;
}

function showPreviousResults(historyId) {
    // Find the history item
    const historyItem = searchHistory.find(item => item.id === historyId);
    if (!historyItem) {
        console.error('History item not found:', historyId);
        return;
    }
    
    // Show results panel if it's hidden
    if (!resultsPanel.classList.contains('visible')) {
        resultsPanel.style.display = 'block';
        setTimeout(() => {
            resultsPanel.classList.add('visible');
            if (mainContent) {
                mainContent.classList.add('with-results');
            }
            const hero = document.querySelector('.hero');
            if (hero) {
                hero.classList.add('with-results');
            }
        }, 10);
    }
    
    // Update results count
    const totalResults = historyItem.resultsCount + historyItem.webSourcesCount;
    resultsCount.textContent = totalResults > 0 
        ? `${totalResults} results retrieved from history`
        : 'No results found';
    
    // Display the stored results
    let resultsHtml = '';
    
    // Show datasets if any
    if (historyItem.results && historyItem.results.length > 0) {
        resultsHtml += `
            <div class="datasets-section">
                <h4 style="color: #3b82f6; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
                    📊 <span>Dataset Results (from ${formatTimestamp(historyItem.timestamp)})</span>
                </h4>
                ${historyItem.results.map((result, index) => createDatasetCard(result, index)).join('')}
            </div>
        `;
    }
    
    // Show web sources if any
    if (historyItem.webSources && historyItem.webSources.length > 0) {
        resultsHtml += `
            <div class="web-sources-section">
                <h4 style="color: #10b981; margin: 0 0 20px 0; display: flex; align-items: center; gap: 10px;">
                    🌐 <span>Web Sources (from ${formatTimestamp(historyItem.timestamp)})</span>
                </h4>
                ${historyItem.webSources.map((source, index) => createWebSourceCard(source, index)).join('')}
            </div>
        `;
    }
    
    if (!resultsHtml) {
        resultsHtml = `
            <div class="no-results-panel">
                <div class="no-results-content">
                    <h3 style="color: #f59e0b; margin-bottom: 15px;">📂 No detailed results stored</h3>
                    <p style="color: #64748b;">This search was performed before result storage was enabled.</p>
                </div>
            </div>
        `;
    }
    
    resultsArea.innerHTML = resultsHtml;
    resultsArea.scrollTop = 0;
    
    // Add visual indicator that these are retrieved results
    const indicator = document.createElement('div');
    indicator.innerHTML = `
        <div style="background: #eff6ff; border: 1px solid #3b82f6; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <div style="color: #1e40af; font-weight: 500; font-size: 0.9rem;">
                📚 Retrieved from search history: "${historyItem.userQuery}"
            </div>
            <div style="color: #64748b; font-size: 0.8rem;">
                Searched ${formatTimestamp(historyItem.timestamp)}
            </div>
        </div>
    `;
    resultsArea.insertBefore(indicator, resultsArea.firstChild);
}

// Global functions for quick action buttons
window.performSearch = performSearch;
window.showPreviousResults = showPreviousResults;
window.clearChatHistory = clearChatHistory;

function closeResultsPanel() {
    if (resultsPanel) {
        resultsPanel.classList.remove('visible');
        if (mainContent) {
            mainContent.classList.remove('with-results');
        }
        // Also reset header
        const hero = document.querySelector('.hero');
        if (hero) {
            hero.classList.remove('with-results');
        }
        setTimeout(() => {
            resultsPanel.style.display = 'none';
        }, 400);
    }
}

function generateRecommendationExplanation(recommendation, dataset, confidence) {
    const title = dataset.title || '';
    const description = dataset.description || '';
    const category = dataset.category || '';
    
    // Generate contextual explanation based on content
    let reasons = [];
    
    // High confidence explanations
    if (confidence > 0.8) {
        reasons.push('Strong keyword match with your search');
    } else if (confidence > 0.6) {
        reasons.push('Good semantic similarity to your query');
    } else {
        reasons.push('Related topic that might be useful');
    }
    
    // Category-specific explanations
    if (category.toLowerCase().includes('transport')) {
        reasons.push('transportation and mobility data');
    } else if (category.toLowerCase().includes('housing')) {
        reasons.push('housing and property information');
    } else if (category.toLowerCase().includes('education')) {
        reasons.push('educational statistics and data');
    }
    
    // Source-specific trust indicators
    if (dataset.agency && dataset.agency.includes('Statistics')) {
        reasons.push('official government statistics');
    } else if (dataset.source && dataset.source.includes('gov.sg')) {
        reasons.push('verified government data source');
    }
    
    return reasons.length > 0 ? reasons.join(', ') : 'Matches your search criteria';
}

async function provideFeedback(datasetId, feedbackType) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                feedback_type: feedbackType,
                query: searchInput.value,
                timestamp: new Date().toISOString()
            })
        });
        
        if (response.ok) {
            // Visual feedback
            const button = event.target;
            const originalText = button.innerHTML;
            button.innerHTML = feedbackType === 'helpful' ? '✓' : '✗';
            button.style.opacity = '0.6';
            button.disabled = true;
            
            setTimeout(() => {
                button.innerHTML = originalText;
                button.style.opacity = '1';
                button.disabled = false;
            }, 2000);
        }
    } catch (error) {
        console.log('Feedback not sent:', error);
    }
}

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

function formatTimestamp(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString('en-SG', {
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return '';
    }
}

function createWebSourceCard(source, index) {
    const title = source.title || 'Untitled Source';
    const url = source.url || '#';
    const description = source.description || 'No description available';
    const sourceType = source.source || 'web';
    const type = source.type || 'general';
    const relevanceScore = source.relevance_score || 0;
    
    // Generate keyword tags based on user query and source content
    const keywordTags = generateKeywordTags(source, searchInput.value);
    
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
                    <div style="margin-left: auto; display: flex; gap: 4px; flex-wrap: wrap;">
                        ${keywordTags.map(tag => `
                            <span onclick="performSearch('${tag}')" style="
                                background: #3b82f6; 
                                color: white; 
                                padding: 2px 8px; 
                                border-radius: 12px; 
                                font-size: 0.75em; 
                                font-weight: 500;
                                cursor: pointer;
                                transition: background 0.2s ease;
                            " onmouseover="this.style.background='#2563eb'" 
                               onmouseout="this.style.background='#3b82f6'">
                                ${tag}
                            </span>
                        `).join('')}
                    </div>
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

function generateKeywordTags(source, userQuery) {
    const title = (source.title || '').toLowerCase();
    const description = (source.description || '').toLowerCase();
    const sourceType = (source.source || '').toLowerCase();
    const query = (userQuery || '').toLowerCase();
    
    // Extract keywords from user query (remove common words)
    const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'how', 'are', 'you', 'today', 'data', 'dataset', 'find', 'search'];
    const queryWords = query.split(/\s+/).filter(word => 
        word.length > 2 && !commonWords.includes(word)
    );
    
    // Predefined relevant tags based on content analysis
    const contentTags = [];
    
    // Singapore-specific tags
    if (title.includes('singapore') || description.includes('singapore') || sourceType.includes('gov.sg')) {
        contentTags.push('singapore');
    }
    
    // Domain-specific tags
    const domainKeywords = {
        'housing': ['housing', 'property', 'hdb', 'real estate', 'residential'],
        'transport': ['transport', 'mrt', 'bus', 'traffic', 'mobility', 'lta'],
        'health': ['health', 'healthcare', 'medical', 'hospital', 'clinic'],
        'education': ['education', 'school', 'university', 'student', 'learning'],
        'economy': ['economic', 'gdp', 'trade', 'finance', 'employment'],
        'government': ['government', 'policy', 'public', 'ministry', 'agency']
    };
    
    // Check for domain matches
    Object.keys(domainKeywords).forEach(domain => {
        if (domainKeywords[domain].some(keyword => 
            title.includes(keyword) || description.includes(keyword) || query.includes(keyword)
        )) {
            contentTags.push(domain);
        }
    });
    
    // Add source type tag
    if (sourceType === 'government_portal') contentTags.push('official');
    if (sourceType === 'zenodo') contentTags.push('research');
    if (sourceType === 'figshare') contentTags.push('academic');
    
    // Combine and deduplicate
    const allTags = [...new Set([...queryWords.slice(0, 2), ...contentTags])];
    
    // Return max 4 most relevant tags
    return allTags.slice(0, 4);
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