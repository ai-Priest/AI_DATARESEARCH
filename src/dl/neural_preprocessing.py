"""
Neural Preprocessing Module - Advanced Data Preparation for Deep Learning
Handles sophisticated data preprocessing, feature engineering, and augmentation for neural networks.
"""

import json
import logging
import pickle
import re  # This was missing!
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class NeuralDataPreprocessor:
    """Advanced data preprocessing for neural network training."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get("data_processing", {})
        self.neural_config = self.data_config.get("neural_preprocessing", {})

        # Initialize tokenizer for text processing
        text_config = self.neural_config.get("text_processing", {})
        self.tokenizer_name = text_config.get("tokenization", "bert-base-uncased")
        self.max_length = text_config.get("max_length", 512)

        # Device setup
        self.device = self._setup_device()
        
        # Initialize BERT model for embeddings along with tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.text_encoder = AutoModel.from_pretrained(self.tokenizer_name)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()  # Set to evaluation mode
            logger.info(f"✅ Loaded tokenizer and encoder: {self.tokenizer_name} on {self.device}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tokenizer {self.tokenizer_name}: {e}")
            # Fallback to simpler model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )
                self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
                self.text_encoder.to(self.device)
                self.text_encoder.eval()
                logger.info(f"✅ Loaded fallback tokenizer: distilbert-base-uncased on {self.device}")
            except:
                self.tokenizer = None
                self.text_encoder = None
                logger.error("❌ Could not load any text encoder")

        # Initialize scalers and encoders
        self.scalers = {}
        self.encoders = {}
        self.feature_projector = None  # For aligning feature dimensions

        # Graph construction for GNN
        self.dataset_graph = None

        logger.info("🧠 NeuralDataPreprocessor initialized")

    def _setup_device(self) -> torch.device:
        """Setup compute device for preprocessing."""
        training_config = self.config.get("training", {})
        device_config = training_config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"🔥 Using CUDA for preprocessing")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS for preprocessing")
            else:
                device = torch.device("cpu")
                logger.info("💻 Using CPU for preprocessing")
        else:
            device = torch.device(device_config)
            logger.info(f"📱 Using specified device for preprocessing: {device}")
        
        return device

    def process_complete_pipeline(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute complete neural preprocessing pipeline."""
        logger.info("🚀 Starting neural preprocessing pipeline")

        try:
            # Load input data
            datasets_df, user_behavior_df, ml_embeddings = self._load_input_data()

            # Process text data for neural networks
            text_features = self._process_text_features(datasets_df)

            # Engineer advanced features
            engineered_features = self._engineer_neural_features(
                datasets_df, user_behavior_df
            )

            # Construct graph for GNN
            graph_data = self._construct_dataset_graph(datasets_df, ml_embeddings)

            # Process user behavior for neural training
            user_features = self._process_user_behavior(user_behavior_df)

            # Create training/validation splits
            train_data, val_data, test_data = self._create_data_splits(
                datasets_df, text_features, engineered_features, user_features
            )

            # Data augmentation
            if self.neural_config.get("augmentation", {}).get("enabled", False):
                train_data = self._apply_data_augmentation(train_data)

            # Prepare final data structure
            processed_data = {
                "train": train_data,
                "validation": val_data,
                "test": test_data,
                "graph": graph_data,
                "metadata": {
                    "num_datasets": len(datasets_df),
                    "num_users": len(user_behavior_df["DEVICE_ID"].unique())
                    if not user_behavior_df.empty
                    and "DEVICE_ID" in user_behavior_df.columns
                    else 0,
                    "text_vocab_size": len(self.tokenizer.vocab)
                    if self.tokenizer
                    else 0,
                    "feature_dimensions": {
                        "text": text_features["input_ids"].shape[1]
                        if "input_ids" in text_features
                        else 0,
                        "engineered": engineered_features.get(
                            "original_features", np.array([])
                        ).shape[1]
                        if engineered_features
                        else 0,
                        "graph_nodes": graph_data["num_nodes"] if graph_data else 0,
                    },
                },
            }

            # Validation
            validation_results = self._validate_processed_data(processed_data)

            logger.info("✅ Neural preprocessing pipeline completed successfully")
            return processed_data, validation_results

        except Exception as e:
            logger.error(f"❌ Neural preprocessing failed: {e}")
            raise

    def _load_input_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Load all input data sources."""
        logger.info("📊 Loading input data sources")

        paths = self.data_config.get("input_paths", {})

        # Load enhanced datasets
        datasets_path = paths.get(
            "enhanced_datasets", "models/datasets_with_ml_quality.csv"
        )
        datasets_df = pd.read_csv(datasets_path)
        logger.info(f"📈 Loaded {len(datasets_df)} enhanced datasets")

        # Load user behavior
        user_behavior_path = paths.get("user_behavior", "data/raw/user_behaviour.csv")
        try:
            user_behavior_df = pd.read_csv(user_behavior_path)
            logger.info(f"👥 Loaded {len(user_behavior_df)} user behavior records")
        except Exception as e:
            logger.warning(f"⚠️ Could not load user behavior data: {e}")
            user_behavior_df = pd.DataFrame()

        # Load ML embeddings
        ml_embeddings_path = paths.get(
            "ml_embeddings", "models/semantic_embeddings.npz"
        )
        try:
            ml_embeddings = np.load(ml_embeddings_path)["embeddings"]
            logger.info(f"🧮 Loaded ML embeddings: {ml_embeddings.shape}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load ML embeddings: {e}")
            ml_embeddings = None

        return datasets_df, user_behavior_df, ml_embeddings

    def _process_text_features(
        self, datasets_df: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Process text data for neural networks - FIXED to generate real embeddings."""
        logger.info("📝 Processing text features for neural networks")

        if not self.tokenizer or not self.text_encoder:
            logger.warning(
                "⚠️ No tokenizer/encoder available, creating random embeddings"
            )
            # Create random 768D embeddings as fallback
            num_samples = len(datasets_df)
            return {
                "text_embeddings": torch.randn(num_samples, 768),
                "input_ids": torch.zeros(
                    num_samples, self.max_length, dtype=torch.long
                ),
                "attention_mask": torch.ones(
                    num_samples, self.max_length, dtype=torch.long
                ),
            }

        # Combine text fields
        text_data = []
        for _, row in datasets_df.iterrows():
            combined_text = f"{row.get('title', '')} {row.get('description', '')} {row.get('tags', '')}"
            text_data.append(combined_text.strip())

        # Tokenize
        text_config = self.neural_config.get("text_processing", {})
        tokenized = self.tokenizer(
            text_data,
            max_length=text_config.get("max_length", 512),
            padding=text_config.get("padding", "max_length"),
            truncation=text_config.get("truncation", True),
            return_tensors="pt",
            return_attention_mask=text_config.get("return_attention_mask", True),
        )

        # Generate actual text embeddings using BERT
        text_embeddings = []
        batch_size = 8  # Process in batches to avoid memory issues

        with torch.no_grad():
            for i in range(0, len(text_data), batch_size):
                batch_input_ids = tokenized["input_ids"][i : i + batch_size].to(self.device)
                batch_attention_mask = tokenized["attention_mask"][i : i + batch_size].to(self.device)

                # Get BERT embeddings
                outputs = self.text_encoder(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )

                # Use CLS token embeddings (first token)
                batch_embeddings = outputs.last_hidden_state[
                    :, 0, :
                ]  # Shape: [batch, 768]
                text_embeddings.append(batch_embeddings.cpu())  # Move back to CPU for concatenation

        # Concatenate all embeddings
        text_embeddings = torch.cat(text_embeddings, dim=0)  # Shape: [num_samples, 768]

        logger.info(f"✅ Generated text embeddings: {text_embeddings.shape}")

        # Return both tokenized inputs and embeddings
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "text_embeddings": text_embeddings,  # NEW: actual 768D embeddings
        }

        return result
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string into embeddings.
        
        Args:
            text: Input text to encode
            
        Returns:
            Numpy array of text embeddings
        """
        if not self.tokenizer or not self.text_encoder:
            logger.warning("No tokenizer/encoder available, returning random embedding")
            return np.random.randn(768)
        
        try:
            # Tokenize the text
            tokenized = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Generate embeddings
            with torch.no_grad():
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)
                
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use pooled output (CLS token) for sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return embeddings.squeeze()
                
        except Exception as e:
            logger.warning(f"Error encoding text: {e}, returning random embedding")
            return np.random.randn(768)

    def _create_feature_projector(
        self, input_dim: int, output_dim: int = 768
    ) -> nn.Module:
        """Create a projector to align feature dimensions."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _engineer_neural_features(
        self, datasets_df: pd.DataFrame, user_behavior_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Engineer advanced features for neural networks - FIXED dimension alignment."""
        logger.info("🔧 Engineering neural features")

        feature_config = self.neural_config.get("feature_engineering", {})
        features = []

        # Basic metadata features
        if "quality_score" in datasets_df.columns:
            features.append(datasets_df["quality_score"].values.reshape(-1, 1))

        # Temporal features
        if feature_config.get("temporal_features", False):
            temporal_features = self._extract_temporal_features(datasets_df)
            if temporal_features is not None:
                features.append(temporal_features)

        # Category encoding
        if feature_config.get("category_encoding", False):
            category_features = self._encode_categories(datasets_df)
            if category_features is not None:
                features.append(category_features)

        # User interaction features
        if not user_behavior_df.empty and feature_config.get(
            "interaction_history", False
        ):
            interaction_features = self._extract_interaction_features(
                datasets_df, user_behavior_df
            )
            if interaction_features is not None:
                features.append(interaction_features)

        if features:
            engineered_features = np.concatenate(features, axis=1)

            # Normalize features
            scaler = StandardScaler()
            engineered_features = scaler.fit_transform(engineered_features)
            self.scalers["neural_features"] = scaler

            # Create projector to align dimensions
            feature_dim = engineered_features.shape[1]
            logger.info(f"📏 Original feature dimension: {feature_dim}")

            # Initialize projector if not exists
            if self.feature_projector is None:
                self.feature_projector = self._create_feature_projector(
                    feature_dim, 768
                )

            # Convert to tensor and project to 768D
            engineered_features_tensor = torch.tensor(
                engineered_features, dtype=torch.float32
            )
            projected_features = self.feature_projector(engineered_features_tensor)

            logger.info(
                f"✅ Engineered features shape: {engineered_features.shape} → Projected: {projected_features.shape}"
            )

            return {
                "original_features": engineered_features,
                "projected_features": projected_features.detach().numpy(),  # 768D aligned features
                "feature_dim": feature_dim,
            }

        logger.warning("⚠️ No features could be engineered")
        return None

    def _extract_temporal_features(
        self, datasets_df: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract temporal features from dataset descriptions."""
        temporal_features = []

        for _, row in datasets_df.iterrows():
            description = str(row.get("description", "")).lower()

            # Extract years mentioned
            years = []
            year_matches = re.findall(r"\b(19|20)\d{2}\b", description)
            if year_matches:
                years = [int(match) for match in year_matches]

            # Temporal indicators
            current_year = datetime.now().year
            features = [
                len(years),  # Number of years mentioned
                max(years) if years else current_year,  # Latest year
                min(years) if years else current_year,  # Earliest year
                1
                if "real-time" in description or "live" in description
                else 0,  # Real-time indicator
                1 if "historical" in description else 0,  # Historical indicator
                1 if "monthly" in description else 0,  # Monthly updates
                1 if "daily" in description else 0,  # Daily updates
                1 if "annual" in description else 0,  # Annual updates
            ]

            temporal_features.append(features)

        return np.array(temporal_features) if temporal_features else None

    def _encode_categories(self, datasets_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Encode categorical features."""
        if "category" not in datasets_df.columns:
            return None

        # Use label encoding for categories
        encoder = LabelEncoder()
        categories = datasets_df["category"].fillna("unknown")
        encoded_categories = encoder.fit_transform(categories)
        self.encoders["category"] = encoder

        # One-hot encoding
        num_categories = len(encoder.classes_)
        one_hot = np.zeros((len(encoded_categories), num_categories))
        one_hot[np.arange(len(encoded_categories)), encoded_categories] = 1

        return one_hot

    def _extract_interaction_features(
        self, datasets_df: pd.DataFrame, user_behavior_df: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract user interaction features for each dataset."""
        if user_behavior_df.empty:
            return None

        # Create dataset interaction summary
        interaction_features = []

        for _, dataset in datasets_df.iterrows():
            dataset_title = dataset.get("title", "")

            # Count interactions for this dataset (simplified for demo)
            features = [
                0,  # Number of related interactions
                0,  # Unique users
                0,  # Average rating
                0,  # Has interactions
            ]

            interaction_features.append(features)

        return np.array(interaction_features) if interaction_features else None

    def _construct_dataset_graph(
        self, datasets_df: pd.DataFrame, ml_embeddings: Optional[np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """Construct graph structure for Graph Neural Networks."""
        logger.info("🕸️ Constructing dataset graph for GNN")

        graph_config = self.config.get("models", {}).get("graph_neural", {})
        if not graph_config.get("enabled", False):
            return None

        try:
            # Create graph
            G = nx.Graph()

            # Add nodes (datasets)
            for idx, row in datasets_df.iterrows():
                G.add_node(
                    idx,
                    title=row.get("title", ""),
                    category=row.get("category", "unknown"),
                    quality_score=row.get("quality_score", 0),
                )

            # Add edges based on similarity
            if ml_embeddings is not None:
                construction_config = graph_config.get("graph_construction", {})
                similarity_threshold = construction_config.get(
                    "similarity_threshold", 0.3
                )
                max_neighbors = construction_config.get("max_neighbors", 10)

                # Compute pairwise similarities
                from sklearn.metrics.pairwise import cosine_similarity

                similarities = cosine_similarity(ml_embeddings)

                # Add edges
                for i in range(len(similarities)):
                    # Get top similar datasets
                    similar_indices = np.argsort(similarities[i])[::-1][
                        1 : max_neighbors + 1
                    ]

                    for j in similar_indices:
                        if similarities[i, j] > similarity_threshold:
                            G.add_edge(i, j, weight=similarities[i, j])

            # Convert to format suitable for PyTorch Geometric
            edge_index = []
            edge_attr = []

            for edge in G.edges(data=True):
                edge_index.append([edge[0], edge[1]])
                edge_index.append([edge[1], edge[0]])  # Undirected graph
                edge_attr.extend([edge[2].get("weight", 1.0)] * 2)

            graph_data = {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "edge_index": torch.tensor(edge_index).t().contiguous()
                if edge_index
                else torch.empty((2, 0), dtype=torch.long),
                "edge_attr": torch.tensor(edge_attr)
                if edge_attr
                else torch.empty((0,)),
                "node_features": self._extract_node_features(datasets_df, G),
            }

            logger.info(
                f"✅ Constructed graph: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges"
            )
            return graph_data

        except Exception as e:
            logger.error(f"❌ Graph construction failed: {e}")
            return None

    def _extract_node_features(
        self, datasets_df: pd.DataFrame, G: nx.Graph
    ) -> torch.Tensor:
        """Extract node features for graph neural network."""
        node_features = []

        for node in G.nodes():
            row = datasets_df.iloc[node]

            # Basic features
            features = [
                row.get("quality_score", 0),
                len(str(row.get("description", ""))),  # Description length
                1
                if "singapore" in str(row.get("title", "")).lower()
                else 0,  # Singapore dataset
                G.degree(node),  # Node degree
            ]

            # Category features (simplified)
            category = str(row.get("category", "unknown")).lower()
            category_features = [
                1 if "housing" in category else 0,
                1 if "transport" in category else 0,
                1 if "environment" in category else 0,
                1 if "population" in category else 0,
                1 if "government" in category else 0,
            ]

            features.extend(category_features)
            node_features.append(features)

        return torch.tensor(node_features, dtype=torch.float32)

    def _process_user_behavior(
        self, user_behavior_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Process user behavior data for neural training."""
        if user_behavior_df.empty:
            return None

        logger.info("👥 Processing user behavior for neural training")

        # Simplified user features
        user_features = {
            "num_users": user_behavior_df.get("DEVICE_ID", pd.Series()).nunique()
            if "DEVICE_ID" in user_behavior_df.columns
            else 0,
            "num_sessions": user_behavior_df.get("SESSION_ID", pd.Series()).nunique()
            if "SESSION_ID" in user_behavior_df.columns
            else 0,
            "total_events": len(user_behavior_df),
        }

        logger.info(
            f"✅ Processed user behavior: {user_features['num_users']} users, {user_features['total_events']} events"
        )
        return user_features

    def _create_data_splits(
        self,
        datasets_df: pd.DataFrame,
        text_features: Dict[str, torch.Tensor],
        engineered_features: Optional[Dict[str, Any]],
        user_features: Optional[Dict[str, Any]],
    ) -> Tuple[Dict, Dict, Dict]:
        """Create train/validation/test splits - FIXED to include all necessary data."""
        logger.info("🔀 Creating data splits")

        # Create indices for splitting
        indices = np.arange(len(datasets_df))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        def split_data(data, indices):
            if isinstance(data, torch.Tensor):
                return data[indices]
            elif isinstance(data, np.ndarray):
                return data[indices]
            elif isinstance(data, dict):
                return {
                    k: split_data(v, indices) if hasattr(v, "__getitem__") else v
                    for k, v in data.items()
                }
            else:
                return data

        # Split datasets
        train_data = {
            "datasets": datasets_df.iloc[train_idx].reset_index(drop=True),
            "indices": train_idx,
        }
        val_data = {
            "datasets": datasets_df.iloc[val_idx].reset_index(drop=True),
            "indices": val_idx,
        }
        test_data = {
            "datasets": datasets_df.iloc[test_idx].reset_index(drop=True),
            "indices": test_idx,
        }

        # Add text features if available
        if text_features:
            train_data["text"] = split_data(text_features, train_idx)
            val_data["text"] = split_data(text_features, val_idx)
            test_data["text"] = split_data(text_features, test_idx)

        # Add engineered features if available - FIXED to use projected features
        if engineered_features is not None:
            train_data["features"] = engineered_features["original_features"][train_idx]
            train_data["projected_features"] = engineered_features[
                "projected_features"
            ][train_idx]
            val_data["features"] = engineered_features["original_features"][val_idx]
            val_data["projected_features"] = engineered_features["projected_features"][
                val_idx
            ]
            test_data["features"] = engineered_features["original_features"][test_idx]
            test_data["projected_features"] = engineered_features["projected_features"][
                test_idx
            ]

        # User features remain global (not split by dataset)
        if user_features:
            train_data["user_features"] = user_features
            val_data["user_features"] = user_features
            test_data["user_features"] = user_features

        logger.info(
            f"✅ Created splits: train({len(train_idx)}), val({len(val_idx)}), test({len(test_idx)})"
        )
        return train_data, val_data, test_data

    def _apply_data_augmentation(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation techniques."""
        logger.info("🔄 Applying data augmentation")

        augmentation_config = self.neural_config.get("augmentation", {})

        # Simple augmentation: add noise to numerical features
        if "features" in train_data and train_data["features"] is not None:
            features = train_data["features"]
            noise_scale = 0.01
            noise = np.random.normal(0, noise_scale, features.shape)
            augmented_features = features + noise

            # Concatenate original and augmented
            train_data["features"] = np.vstack([features, augmented_features])

            # Duplicate other data accordingly
            if "datasets" in train_data:
                train_data["datasets"] = pd.concat(
                    [train_data["datasets"], train_data["datasets"]]
                ).reset_index(drop=True)

            # Also duplicate text features
            if "text" in train_data:
                for key in train_data["text"]:
                    if isinstance(train_data["text"][key], torch.Tensor):
                        train_data["text"][key] = torch.cat(
                            [train_data["text"][key], train_data["text"][key]]
                        )

            # Duplicate projected features
            if "projected_features" in train_data:
                train_data["projected_features"] = np.vstack(
                    [train_data["projected_features"], train_data["projected_features"]]
                )

            logger.info(f"✅ Augmented features: {train_data['features'].shape}")

        return train_data

    def _validate_processed_data(
        self, processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the processed data."""
        logger.info("✅ Validating processed data")

        validation_results = {
            "status": "PASS",
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check data availability
        validation_results["checks"]["has_train_data"] = "train" in processed_data
        validation_results["checks"]["has_validation_data"] = (
            "validation" in processed_data
        )
        validation_results["checks"]["has_test_data"] = "test" in processed_data

        # Check data sizes
        if "train" in processed_data:
            train_size = len(processed_data["train"].get("datasets", []))
            validation_results["checks"]["train_size"] = train_size

            if train_size < 50:
                validation_results["warnings"].append(
                    f"Small training set: {train_size} samples"
                )

        # Check feature dimensions
        metadata = processed_data.get("metadata", {})
        feature_dims = metadata.get("feature_dimensions", {})

        for feature_type, dim in feature_dims.items():
            validation_results["checks"][f"{feature_type}_dimension"] = dim
            if dim == 0:
                validation_results["warnings"].append(
                    f"No {feature_type} features available"
                )

        # Overall status
        if validation_results["errors"]:
            validation_results["status"] = "FAIL"
        elif validation_results["warnings"]:
            validation_results["status"] = "PASS_WITH_WARNINGS"

        logger.info(f"✅ Data validation: {validation_results['status']}")
        return validation_results


class NeuralDataset(Dataset):
    """PyTorch Dataset for neural network training."""

    def __init__(self, data: Dict[str, Any], mode: str = "train"):
        self.data = data
        self.mode = mode
        self.datasets_df = data.get("datasets", pd.DataFrame())

        # Extract features
        self.text_features = data.get("text", {})
        self.engineered_features = data.get("features")
        self.projected_features = data.get(
            "projected_features"
        )  # NEW: 768D aligned features
        self.user_features = data.get("user_features", {})

    def __len__(self):
        return len(self.datasets_df)

    def __getitem__(self, idx):
        item = {"idx": torch.tensor(idx, dtype=torch.long)}

        # Add text features - both tokens and embeddings
        if self.text_features:
            if "input_ids" in self.text_features:
                item["text_input_ids"] = self.text_features["input_ids"][idx]
            if "attention_mask" in self.text_features:
                item["text_attention_mask"] = self.text_features["attention_mask"][idx]
            if "text_embeddings" in self.text_features:
                item["text_embeddings"] = self.text_features["text_embeddings"][idx]

        # Add engineered features (original 21D)
        if self.engineered_features is not None:
            item["features"] = torch.tensor(
                self.engineered_features[idx], dtype=torch.float32
            )

        # Add projected features (768D aligned)
        if self.projected_features is not None:
            item["projected_features"] = torch.tensor(
                self.projected_features[idx], dtype=torch.float32
            )

        # Add labels (for supervised training)
        item["labels"] = torch.tensor(1.0, dtype=torch.float32)

        return item


def create_neural_data_preprocessor(config: Dict) -> NeuralDataPreprocessor:
    """Factory function to create neural data preprocessor."""
    return NeuralDataPreprocessor(config)


def demo_neural_preprocessing():
    """Demonstrate neural preprocessing capabilities."""
    print("🧠 Neural Preprocessing Demo")

    # Create mock config
    config = {
        "data_processing": {
            "neural_preprocessing": {
                "text_processing": {
                    "tokenization": "bert-base-uncased",
                    "max_length": 256,
                },
                "feature_engineering": {
                    "temporal_features": True,
                    "category_encoding": True,
                },
                "augmentation": {"enabled": True},
            }
        },
        "models": {"graph_neural": {"enabled": True}},
    }

    preprocessor = create_neural_data_preprocessor(config)
    print("✅ Neural preprocessor created successfully")

    # Note: Full demo would require actual data files
    print("💡 Ready for neural network training!")


if __name__ == "__main__":
    demo_neural_preprocessing()
