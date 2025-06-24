"""
Domain-Specific Fine-tuning for Singapore Government Dataset Search
Enhances semantic models with domain knowledge for better recommendations.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

class DomainFineTuner:
    """Fine-tune semantic models on Singapore government dataset domain."""
    
    def __init__(self, base_model: str = "all-mpnet-base-v2"):
        """
        Initialize with base model for fine-tuning.
        
        Args:
            base_model: Base sentence transformer model to fine-tune
        """
        self.base_model = base_model
        self.model = None
        self.domain_examples = []
        self.evaluation_data = []
        
    def load_datasets(self, singapore_path: str, global_path: str) -> pd.DataFrame:
        """Load and combine datasets for fine-tuning."""
        logger.info("📊 Loading datasets for domain fine-tuning")
        
        singapore_df = pd.read_csv(singapore_path)
        global_df = pd.read_csv(global_path)
        
        # Combine datasets
        combined_df = pd.concat([singapore_df, global_df], ignore_index=True)
        
        logger.info(f"✅ Loaded {len(combined_df)} datasets for fine-tuning")
        return combined_df
    
    def create_domain_training_pairs(self, df: pd.DataFrame) -> List[InputExample]:
        """
        Create training pairs from dataset relationships and similarities.
        
        Strategy:
        1. Similar datasets (high similarity) → positive pairs
        2. Different categories → negative pairs  
        3. Keyword-based relationships → positive pairs
        4. Quality-based relationships → positive pairs
        """
        logger.info("🔄 Creating domain-specific training pairs")
        
        training_examples = []
        
        # Create text representations
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['description'].fillna('') + ' ' + 
            df['tags'].fillna('') + ' ' + 
            df['category'].fillna('')
        )
        
        # Strategy 1: Category-based positive pairs
        categories = df['category'].value_counts()
        for category in categories.index[:10]:  # Top 10 categories
            category_datasets = df[df['category'] == category]
            if len(category_datasets) >= 2:
                # Create positive pairs within same category
                for i in range(min(5, len(category_datasets))):
                    for j in range(i + 1, min(i + 3, len(category_datasets))):
                        text1 = category_datasets.iloc[i]['combined_text']
                        text2 = category_datasets.iloc[j]['combined_text']
                        training_examples.append(InputExample(
                            texts=[text1, text2], 
                            label=0.8  # High similarity for same category
                        ))
        
        # Strategy 2: Keyword-based positive pairs
        # Find datasets with overlapping keywords in tags
        for idx1, row1 in df.iterrows():
            if pd.isna(row1['tags']):
                continue
            tags1 = set(str(row1['tags']).lower().split())
            
            for idx2, row2 in df.iterrows():
                if idx1 >= idx2 or pd.isna(row2['tags']):
                    continue
                    
                tags2 = set(str(row2['tags']).lower().split())
                overlap = len(tags1.intersection(tags2))
                
                if overlap >= 2:  # At least 2 shared keywords
                    similarity_score = min(0.9, 0.6 + (overlap * 0.1))
                    training_examples.append(InputExample(
                        texts=[row1['combined_text'], row2['combined_text']],
                        label=similarity_score
                    ))
                    
                    if len(training_examples) >= 1000:  # Limit for efficiency
                        break
            if len(training_examples) >= 1000:
                break
        
        # Strategy 3: Quality-based relationships
        high_quality = df[df['quality_score'] > 0.8]
        medium_quality = df[(df['quality_score'] > 0.6) & (df['quality_score'] <= 0.8)]
        
        # High quality datasets should be more similar to each other
        for i in range(min(10, len(high_quality))):
            for j in range(i + 1, min(i + 3, len(high_quality))):
                training_examples.append(InputExample(
                    texts=[high_quality.iloc[i]['combined_text'], 
                          high_quality.iloc[j]['combined_text']],
                    label=0.7
                ))
        
        # Strategy 4: Negative pairs (different domains)
        singapore_datasets = df[df['source'].str.contains('singapore|sg', case=False, na=False)]
        global_datasets = df[~df['source'].str.contains('singapore|sg', case=False, na=False)]
        
        # Create some negative pairs between very different domains
        for i in range(min(50, len(singapore_datasets))):
            for j in range(min(2, len(global_datasets))):
                sg_text = singapore_datasets.iloc[i]['combined_text']
                global_text = global_datasets.iloc[j]['combined_text']
                
                # Only create negative pairs if categories are very different
                sg_category = str(singapore_datasets.iloc[i].get('category', '')).lower()
                global_category = str(global_datasets.iloc[j].get('category', '')).lower()
                
                if sg_category != global_category:
                    training_examples.append(InputExample(
                        texts=[sg_text, global_text],
                        label=0.2  # Low similarity for different domains
                    ))
        
        logger.info(f"✅ Created {len(training_examples)} training pairs")
        self.domain_examples = training_examples
        return training_examples
    
    def create_evaluation_data(self, df: pd.DataFrame) -> List[InputExample]:
        """Create evaluation dataset from ground truth scenarios."""
        logger.info("🎯 Creating evaluation dataset")
        
        evaluation_examples = []
        
        # Use a subset of data for evaluation
        eval_df = df.sample(n=min(50, len(df)), random_state=42)
        
        for idx, row in eval_df.iterrows():
            # Create positive example (dataset with itself - should be 1.0)
            evaluation_examples.append(InputExample(
                texts=[row['combined_text'], row['combined_text']],
                label=1.0
            ))
            
            # Create some moderate similarity examples
            same_category = df[df['category'] == row['category']]
            if len(same_category) > 1:
                other_in_category = same_category[same_category.index != idx].iloc[0]
                evaluation_examples.append(InputExample(
                    texts=[row['combined_text'], other_in_category['combined_text']],
                    label=0.7
                ))
        
        logger.info(f"✅ Created {len(evaluation_examples)} evaluation examples")
        self.evaluation_data = evaluation_examples
        return evaluation_examples
    
    def fine_tune_model(self, output_path: str = "models/domain_tuned_model", 
                       epochs: int = 3, batch_size: int = 16) -> SentenceTransformer:
        """
        Fine-tune the model on domain-specific data.
        
        Args:
            output_path: Path to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Fine-tuned SentenceTransformer model
        """
        logger.info(f"🚀 Starting domain fine-tuning with {self.base_model}")
        
        # Load base model
        self.model = SentenceTransformer(self.base_model)
        
        # Create data loader
        train_dataloader = DataLoader(self.domain_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function (Cosine Similarity Loss)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator if evaluation data exists
        evaluator = None
        if self.evaluation_data:
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                self.evaluation_data, name='domain_eval'
            )
        
        # Fine-tune the model
        logger.info(f"🔄 Training for {epochs} epochs with {len(self.domain_examples)} examples")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=100,
            warmup_steps=100,
            output_path=output_path,
            save_best_model=True
        )
        
        logger.info(f"✅ Fine-tuning complete. Model saved to {output_path}")
        return self.model
    
    def evaluate_improvement(self, original_model_name: str, 
                           test_queries: List[str] = None) -> Dict[str, float]:
        """
        Compare fine-tuned model performance with original model.
        
        Args:
            original_model_name: Name of original model to compare against
            test_queries: Optional test queries for evaluation
            
        Returns:
            Dictionary with comparison metrics
        """
        logger.info("📊 Evaluating fine-tuning improvements")
        
        if test_queries is None:
            test_queries = [
                "singapore housing market data",
                "transport traffic statistics",
                "government budget information",
                "population demographics singapore",
                "environmental data singapore"
            ]
        
        # Load original model for comparison
        original_model = SentenceTransformer(original_model_name)
        
        # Test on some sample dataset texts
        sample_texts = [example.texts[0] for example in self.evaluation_data[:20]]
        
        results = {}
        
        for query in test_queries:
            # Get embeddings from both models
            query_embedding_original = original_model.encode([query])
            query_embedding_finetuned = self.model.encode([query])
            
            # Get embeddings for sample texts
            text_embeddings_original = original_model.encode(sample_texts)
            text_embeddings_finetuned = self.model.encode(sample_texts)
            
            # Calculate similarities
            similarities_original = cosine_similarity(query_embedding_original, text_embeddings_original)[0]
            similarities_finetuned = cosine_similarity(query_embedding_finetuned, text_embeddings_finetuned)[0]
            
            # Calculate improvement metrics
            avg_sim_original = np.mean(similarities_original)
            avg_sim_finetuned = np.mean(similarities_finetuned)
            
            results[f"{query}_improvement"] = (avg_sim_finetuned - avg_sim_original) / avg_sim_original
        
        overall_improvement = np.mean(list(results.values()))
        results['overall_improvement'] = overall_improvement
        
        logger.info(f"✅ Overall improvement: {overall_improvement:.1%}")
        return results
    
    def save_training_info(self, output_path: str):
        """Save training information and statistics."""
        training_info = {
            'base_model': self.base_model,
            'num_training_examples': len(self.domain_examples),
            'num_evaluation_examples': len(self.evaluation_data),
            'training_strategies': [
                'Category-based positive pairs',
                'Keyword-based relationships', 
                'Quality-based relationships',
                'Cross-domain negative pairs'
            ]
        }
        
        with open(f"{output_path}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✅ Training info saved to {output_path}/training_info.json")


def run_domain_fine_tuning():
    """Main function to run domain fine-tuning process."""
    
    # Initialize fine-tuner
    fine_tuner = DomainFineTuner(base_model="all-mpnet-base-v2")
    
    # Load datasets
    df = fine_tuner.load_datasets(
        "data/processed/singapore_datasets.csv",
        "data/processed/global_datasets.csv"
    )
    
    # Create training pairs
    fine_tuner.create_domain_training_pairs(df)
    
    # Create evaluation data
    fine_tuner.create_evaluation_data(df)
    
    # Fine-tune model
    model = fine_tuner.fine_tune_model(
        output_path="models/singapore_domain_model",
        epochs=3,
        batch_size=16
    )
    
    # Evaluate improvement
    improvements = fine_tuner.evaluate_improvement("all-mpnet-base-v2")
    
    # Save training info
    fine_tuner.save_training_info("models/singapore_domain_model")
    
    print("🎉 Domain fine-tuning complete!")
    print(f"📈 Overall improvement: {improvements['overall_improvement']:.1%}")
    
    return model, improvements


if __name__ == "__main__":
    run_domain_fine_tuning()