def _normalize_query_for_source(self, query: str, source: str) -> str:
    """
    Normalize conversational query for specific external sources
    Removes conversational language and extracts core search terms
    Enhanced for AWS Open Data URL generation with proper search parameters
    """
    if not query or not query.strip():
        return ""
    
    normalized = query.lower().strip()
    
    # Remove common conversational patterns in order
    conversational_patterns = [
        # Complex patterns first
        r'^(can you|could you|please)\s+(find me|get me|show me)\s+',
        r'^(i need|i want|i\'m looking for|looking for)\s+',
        r'^(find me|get me|show me|search for)\s+',
        r'^(can you|could you|please)\s+',
        r'^(find|search)\s+',
        # Remove remaining filler words
        r'\b(some|any)\s+',
        r'\s+(related to|regarding|concerning)\s+',
        # Remove trailing words
        r'\s+(please|thanks|thank you)$',
        r'\s+(data|dataset|datasets)$'
    ]
    
    for pattern in conversational_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE).strip()
    
    # Handle "about" separately to preserve spacing
    normalized = re.sub(r'\s+about\s+', ' ', normalized, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Source-specific processing
    if source == 'kaggle':
        # For Kaggle, ensure we have meaningful search terms
        # If the result is too short or generic, keep more of the original
        if len(normalized.split()) < 1 or normalized in ['me', 'it', 'this', 'that']:
            # Extract key nouns from original query
            # Simple noun extraction - look for meaningful words
            words = query.lower().split()
            meaningful_words = []
            skip_words = {'i', 'need', 'want', 'looking', 'for', 'find', 'me', 'get', 'show', 
                         'search', 'can', 'you', 'could', 'please', 'some', 'any', 'about',
                         'data', 'dataset', 'datasets', 'thanks', 'thank', 'you'}
            
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and clean_word not in skip_words and len(clean_word) > 2:
                    meaningful_words.append(clean_word)
            
            if meaningful_words:
                normalized = ' '.join(meaningful_words)
                
    elif source == 'world_bank':
        # World Bank prefers broader economic/development terms
        # Keep country names and economic indicators
        # World Bank search works better with broader terms
        if len(normalized.split()) > 3:
            # For longer queries, focus on key economic/development terms
            key_terms = []
            words = normalized.split()
            economic_keywords = ['gdp', 'economy', 'economic', 'growth', 'development', 'poverty', 
                               'education', 'health', 'population', 'employment', 'trade', 'finance']
            
            for word in words:
                if word in economic_keywords or len(word) > 4:  # Keep longer words and economic terms
                    key_terms.append(word)
            
            if key_terms:
                normalized = ' '.join(key_terms[:3])  # Limit to 3 key terms
                
    elif source == 'aws':
        # AWS Open Data works better with specific domain terms
        # Remove very conversational elements but keep meaningful terms
        if len(normalized.split()) > 4:
            # For longer queries, focus on key domain terms
            key_terms = []
            words = normalized.split()
            domain_keywords = ['climate', 'weather', 'satellite', 'imagery', 'genomics', 'health', 
                             'transportation', 'economic', 'financial', 'machine', 'learning', 
                             'astronomy', 'space', 'geospatial', 'environmental', 'biological']
            
            for word in words:
                if word in domain_keywords or len(word) > 4:  # Keep longer words and domain terms
                    key_terms.append(word)
            
            if key_terms:
                normalized = ' '.join(key_terms[:4])  # Limit to 4 key terms for AWS
    
    return normalized if normalized else query  # Fallback to original if empty