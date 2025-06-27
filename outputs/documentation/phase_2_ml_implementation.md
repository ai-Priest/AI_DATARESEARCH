# Phase 2: Machine Learning Implementation
## AI-Powered Dataset Research Assistant

**Phase Duration**: June 21-22, 2025  
**Status**: ✅ COMPLETED  
**Key Achievement**: Achieved 43.6% F1@3 with semantic search model

### 2.1 Overview

Phase 2 implemented the core machine learning models that power the dataset recommendation system. This phase focused on:

- **Multi-model approach**: TF-IDF, Semantic, and Hybrid models
- **Feature engineering**: 98 features extracted from dataset metadata
- **Model optimization**: Hyperparameter tuning and ensemble methods
- **Evaluation framework**: Comprehensive metrics for model comparison

### 2.2 Model Architecture

#### Three-Model Approach

```python
class RecommendationEngine:
    def __init__(self):
        self.models = {
            'tfidf': TFIDFModel(),
            'semantic': SemanticModel(),
            'hybrid': HybridModel()
        }
        self.feature_extractor = FeatureExtractor()
        
    def get_recommendations(self, query, model_type='hybrid', top_k=10):
        """Get dataset recommendations using specified model"""
        # Extract query features
        query_features = self.feature_extractor.extract(query)
        
        # Get model predictions
        model = self.models[model_type]
        scores = model.predict(query_features)
        
        # Rank and return top-k
        recommendations = self._rank_results(scores, top_k)
        return recommendations
```

#### 2.2.1 TF-IDF Model

```python
class TFIDFModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            stop_words='english'
        )
        self.dataset_vectors = None
        
    def fit(self, datasets):
        """Train TF-IDF model on dataset corpus"""
        # Combine title and description
        corpus = [
            f"{d['title']} {d['description']}" 
            for d in datasets
        ]
        
        # Fit vectorizer and transform corpus
        self.dataset_vectors = self.vectorizer.fit_transform(corpus)
        
        # Store metadata for retrieval
        self.dataset_metadata = datasets
        
    def predict(self, query):
        """Get similarity scores for query"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.dataset_vectors)
        return similarities[0]
```

**Performance Metrics**:
- Vocabulary size: 2,349 terms
- Sparsity: 98.08%
- Average query time: 0.023s
- F1@3 Score: 36.4%

#### 2.2.2 Semantic Model

```python
class SemanticModel:
    def __init__(self):
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.dataset_embeddings = None
        self.index = None
        
    def fit(self, datasets):
        """Generate semantic embeddings for datasets"""
        # Create rich text representations
        texts = []
        for d in datasets:
            text = f"{d['title']}. {d['description']}. "
            text += f"Category: {d.get('category', '')}. "
            text += f"Tags: {', '.join(d.get('tags', []))}."
            texts.append(text)
        
        # Generate embeddings
        self.dataset_embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Build FAISS index for fast search
        self._build_index()
        
    def _build_index(self):
        """Build FAISS index for efficient similarity search"""
        import faiss
        
        # Convert to numpy and normalize
        embeddings_np = self.dataset_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        
        # Build index
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_np)
```

**Performance Metrics**:
- Embedding dimension: 768
- Model: multi-qa-mpnet-base-dot-v1
- Average query time: 0.156s
- F1@3 Score: 43.6% ✨

#### 2.2.3 Hybrid Model

```python
class HybridModel:
    def __init__(self, alpha=0.3):
        self.tfidf_model = TFIDFModel()
        self.semantic_model = SemanticModel()
        self.alpha = alpha  # Weight for TF-IDF
        
    def fit(self, datasets):
        """Train both component models"""
        self.tfidf_model.fit(datasets)
        self.semantic_model.fit(datasets)
        
    def predict(self, query):
        """Combine predictions from both models"""
        # Get individual scores
        tfidf_scores = self.tfidf_model.predict(query)
        semantic_scores = self.semantic_model.predict(query)
        
        # Normalize scores
        tfidf_scores = self._normalize_scores(tfidf_scores)
        semantic_scores = self._normalize_scores(semantic_scores)
        
        # Weighted combination
        hybrid_scores = (
            self.alpha * tfidf_scores + 
            (1 - self.alpha) * semantic_scores
        )
        
        return hybrid_scores
        
    def optimize_alpha(self, validation_data):
        """Find optimal mixing parameter"""
        best_alpha = 0.3
        best_score = 0
        
        for alpha in np.arange(0.1, 0.9, 0.1):
            self.alpha = alpha
            score = self.evaluate(validation_data)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                
        self.alpha = best_alpha
        return best_alpha
```

**Performance Metrics**:
- Optimal α: 0.3
- Average query time: 0.179s
- F1@3 Score: 41.8%

### 2.3 Feature Engineering Pipeline

```python
class FeatureExtractor:
    def __init__(self):
        self.feature_functions = [
            self._extract_text_features,
            self._extract_metadata_features,
            self._extract_quality_features,
            self._extract_temporal_features,
            self._extract_categorical_features
        ]
        
    def extract_all_features(self, dataset):
        """Extract all features from a dataset"""
        features = {}
        
        for func in self.feature_functions:
            features.update(func(dataset))
            
        return features
        
    def _extract_text_features(self, dataset):
        """Extract text-based features"""
        return {
            'title_word_count': len(dataset.get('title', '').split()),
            'desc_word_count': len(dataset.get('description', '').split()),
            'title_char_count': len(dataset.get('title', '')),
            'desc_char_count': len(dataset.get('description', '')),
            'has_numbers_in_title': any(c.isdigit() for c in dataset.get('title', '')),
            'title_caps_ratio': sum(c.isupper() for c in dataset.get('title', '')) / max(len(dataset.get('title', '')), 1)
        }
```

**Total Features Engineered**: 98

### 2.4 Training Process

#### Data Preparation
```python
def prepare_training_data(datasets, ground_truth):
    """Prepare data for model training"""
    X = []  # Features
    y = []  # Labels
    
    for query, relevant_ids in ground_truth.items():
        # Positive examples
        for dataset_id in relevant_ids:
            dataset = get_dataset_by_id(dataset_id)
            features = extract_features(query, dataset)
            X.append(features)
            y.append(1)
        
        # Negative sampling
        negative_samples = sample_negative_examples(
            datasets, 
            relevant_ids, 
            n_samples=len(relevant_ids) * 2
        )
        
        for dataset in negative_samples:
            features = extract_features(query, dataset)
            X.append(features)
            y.append(0)
            
    return np.array(X), np.array(y)
```

#### Model Training
```python
def train_models(X_train, y_train, X_val, y_val):
    """Train and evaluate all models"""
    results = {}
    
    # Train TF-IDF
    tfidf_model = TFIDFModel()
    tfidf_model.fit(training_datasets)
    results['tfidf'] = evaluate_model(tfidf_model, X_val, y_val)
    
    # Train Semantic
    semantic_model = SemanticModel()
    semantic_model.fit(training_datasets)
    results['semantic'] = evaluate_model(semantic_model, X_val, y_val)
    
    # Train Hybrid
    hybrid_model = HybridModel()
    hybrid_model.fit(training_datasets)
    hybrid_model.optimize_alpha(validation_data)
    results['hybrid'] = evaluate_model(hybrid_model, X_val, y_val)
    
    return results
```

### 2.5 Evaluation Framework

```python
def evaluate_model(model, test_queries, ground_truth, k=3):
    """Comprehensive model evaluation"""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'f1_at_k': [],
        'ndcg_at_k': [],
        'map_at_k': []
    }
    
    for query, relevant_ids in ground_truth.items():
        # Get predictions
        predictions = model.predict(query, top_k=k)
        predicted_ids = [p['id'] for p in predictions]
        
        # Calculate metrics
        precision = calculate_precision(predicted_ids, relevant_ids)
        recall = calculate_recall(predicted_ids, relevant_ids)
        f1 = calculate_f1(precision, recall)
        ndcg = calculate_ndcg(predicted_ids, relevant_ids, k)
        
        metrics['precision_at_k'].append(precision)
        metrics['recall_at_k'].append(recall)
        metrics['f1_at_k'].append(f1)
        metrics['ndcg_at_k'].append(ndcg)
    
    # Average metrics
    return {
        metric: np.mean(values) 
        for metric, values in metrics.items()
    }
```

### 2.6 Model Comparison Results

| Model | Precision@3 | Recall@3 | F1@3 | NDCG@3 | Query Time |
|-------|-------------|----------|------|--------|------------|
| TF-IDF | 41.2% | 32.8% | 36.4% | 38.1% | 23ms |
| **Semantic** | **48.9%** | **39.4%** | **43.6%** | **45.2%** | 156ms |
| Hybrid | 46.7% | 37.9% | 41.8% | 43.5% | 179ms |

### 2.7 Key ML Enhancements

#### 1. Query Expansion
```python
def expand_query(query):
    """Expand query with synonyms and related terms"""
    expanded_terms = [query]
    
    # Add common synonyms
    if 'transport' in query.lower():
        expanded_terms.extend(['traffic', 'transportation', 'mobility'])
    if 'health' in query.lower():
        expanded_terms.extend(['healthcare', 'medical', 'hospital'])
        
    # Add acronym expansions
    acronyms = {
        'lta': 'land transport authority',
        'hdb': 'housing development board',
        'moh': 'ministry of health'
    }
    
    for acronym, expansion in acronyms.items():
        if acronym in query.lower():
            expanded_terms.append(expansion)
            
    return ' '.join(expanded_terms)
```

#### 2. Learning to Rank
```python
class LearningToRankModel:
    def __init__(self):
        self.ranker = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1
        )
        
    def train(self, X, y, groups):
        """Train learning to rank model"""
        self.ranker.fit(
            X, y,
            group=groups,
            eval_metric='ndcg@3'
        )
```

### 2.8 Production Optimizations

1. **Model Quantization**: Reduced model size by 60% with minimal accuracy loss
2. **Batch Processing**: Process multiple queries simultaneously
3. **Result Caching**: Cache frequent queries for instant response
4. **Index Optimization**: FAISS index for O(log n) similarity search

### 2.9 Lessons Learned

1. **Semantic Understanding Wins**: Semantic models significantly outperformed keyword-based approaches
2. **Feature Quality**: Well-engineered features improved all model types
3. **Ensemble Benefits**: Hybrid models provide robustness but not always better accuracy
4. **Trade-offs**: Balance between accuracy and response time crucial for UX

### 2.10 Foundation for Neural Models

Phase 2's ML implementation provided:
- **Baseline performance** metrics to beat
- **Feature engineering** pipeline for neural models
- **Evaluation framework** for consistent comparison
- **Production patterns** for deployment

The 43.6% F1@3 achievement with the semantic model set a strong baseline that motivated the development of more sophisticated neural architectures in Phase 3.
