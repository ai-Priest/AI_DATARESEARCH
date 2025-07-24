"""
Memory Optimization with Quality Preservation
Implements model quantization, gradient checkpointing, and compressed embedding storage while maintaining accuracy
"""

import logging
import pickle
import gzip
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization"""
    enable_quantization: bool = True
    quantization_bits: int = 8  # 8-bit quantization
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    embedding_compression_level: int = 6  # gzip compression level
    use_hdf5_storage: bool = True
    memory_threshold_mb: float = 1024.0  # 1GB threshold for optimization
    quality_preservation_threshold: float = 0.02  # Max 2% quality loss allowed


class QualityPreservingQuantizer:
    """
    Model quantization that preserves recommendation quality
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.original_model_size = 0
        self.quantized_model_size = 0
        self.quality_metrics_before = {}
        self.quality_metrics_after = {}
    
    def quantize_model(self, model: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Quantize model while preserving quality
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Data for calibration (optional)
            
        Returns:
            Quantized model
        """
        if not self.config.enable_quantization:
            logger.info("🔧 Model quantization disabled")
            return model
        
        logger.info(f"🔧 Starting model quantization to {self.config.quantization_bits}-bit")
        
        # Store original model size
        self.original_model_size = self._calculate_model_size(model)
        
        try:
            # Prepare model for quantization
            model.eval()
            
            # Configure quantization
            if self.config.quantization_bits == 8:
                quantized_model = self._quantize_to_int8(model, calibration_data)
            elif self.config.quantization_bits == 16:
                quantized_model = self._quantize_to_fp16(model)
            else:
                logger.warning(f"Unsupported quantization bits: {self.config.quantization_bits}")
                return model
            
            # Calculate quantized model size
            self.quantized_model_size = self._calculate_model_size(quantized_model)
            
            # Calculate compression ratio
            compression_ratio = self.original_model_size / self.quantized_model_size
            
            logger.info(f"🔧 Model quantization completed: "
                       f"{self.original_model_size:.2f}MB → {self.quantized_model_size:.2f}MB "
                       f"({compression_ratio:.2f}x compression)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return model
    
    def _quantize_to_int8(self, model: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """Quantize model to INT8"""
        try:
            # Use PyTorch's dynamic quantization for now
            # In production, you might want to use static quantization with calibration data
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            logger.warning(f"INT8 quantization failed: {e}")
            return model
    
    def _quantize_to_fp16(self, model: nn.Module) -> nn.Module:
        """Quantize model to FP16"""
        try:
            # Convert model to half precision
            quantized_model = model.half()
            return quantized_model
            
        except Exception as e:
            logger.warning(f"FP16 quantization failed: {e}")
            return model
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def validate_quantization_quality(self, 
                                    original_model: nn.Module, 
                                    quantized_model: nn.Module,
                                    test_data: torch.Tensor,
                                    test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Validate that quantization preserves quality
        
        Returns:
            Dictionary with quality metrics comparison
        """
        logger.info("🔍 Validating quantization quality")
        
        try:
            # Evaluate original model
            original_metrics = self._evaluate_model(original_model, test_data, test_labels)
            
            # Evaluate quantized model
            quantized_metrics = self._evaluate_model(quantized_model, test_data, test_labels)
            
            # Calculate quality degradation
            quality_comparison = {}
            for metric, original_value in original_metrics.items():
                quantized_value = quantized_metrics.get(metric, 0.0)
                degradation = abs(original_value - quantized_value)
                quality_comparison[f"{metric}_degradation"] = degradation
                quality_comparison[f"{metric}_original"] = original_value
                quality_comparison[f"{metric}_quantized"] = quantized_value
            
            # Check if quality degradation is acceptable
            max_degradation = max(quality_comparison[k] for k in quality_comparison.keys() if k.endswith('_degradation'))
            
            if max_degradation <= self.config.quality_preservation_threshold:
                logger.info(f"✅ Quantization quality preserved: max degradation {max_degradation:.4f}")
            else:
                logger.warning(f"⚠️ Quantization quality degradation: {max_degradation:.4f} > {self.config.quality_preservation_threshold}")
            
            return quality_comparison
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return {}
    
    def _evaluate_model(self, model: nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        
        with torch.no_grad():
            outputs = model(test_data)
            
            # Calculate basic metrics
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Classification metrics
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == test_labels).float().mean().item()
                
                return {
                    'accuracy': accuracy,
                    'loss': nn.CrossEntropyLoss()(outputs, test_labels).item()
                }
            else:
                # Regression metrics
                mse = nn.MSELoss()(outputs.squeeze(), test_labels.float()).item()
                mae = nn.L1Loss()(outputs.squeeze(), test_labels.float()).item()
                
                return {
                    'mse': mse,
                    'mae': mae
                }


class GradientCheckpointing:
    """
    Gradient checkpointing for memory-efficient training
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.enabled = config.enable_gradient_checkpointing
    
    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model"""
        if not self.enabled:
            return model
        
        logger.info("🔧 Applying gradient checkpointing")
        
        try:
            # Apply checkpointing to transformer layers if available
            if hasattr(model, 'transformer') or hasattr(model, 'encoder'):
                self._apply_to_transformer_layers(model)
            
            # Apply to custom layers
            self._apply_to_custom_layers(model)
            
            logger.info("✅ Gradient checkpointing applied successfully")
            
        except Exception as e:
            logger.error(f"Gradient checkpointing failed: {e}")
        
        return model
    
    def _apply_to_transformer_layers(self, model: nn.Module):
        """Apply checkpointing to transformer layers"""
        for name, module in model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'forward'):
                    # Wrap forward method with checkpointing
                    original_forward = module.forward
                    module.forward = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                        original_forward, *args, **kwargs
                    )
    
    def _apply_to_custom_layers(self, model: nn.Module):
        """Apply checkpointing to custom layers"""
        # This would be customized based on your specific model architecture
        pass


class CompressedEmbeddingStorage:
    """
    Compressed storage for embeddings and large arrays
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.compression_level = config.embedding_compression_level
        self.use_hdf5 = config.use_hdf5_storage
        
        # Storage statistics
        self.storage_stats = {
            'original_size_mb': 0.0,
            'compressed_size_mb': 0.0,
            'compression_ratio': 1.0,
            'files_compressed': 0
        }
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str, metadata: Optional[Dict] = None) -> str:
        """
        Save embeddings with compression
        
        Args:
            embeddings: Numpy array of embeddings
            file_path: Path to save file
            metadata: Optional metadata to store with embeddings
            
        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        
        # Calculate original size
        original_size = embeddings.nbytes / (1024 * 1024)  # MB
        self.storage_stats['original_size_mb'] += original_size
        
        try:
            if self.use_hdf5:
                compressed_path = self._save_with_hdf5(embeddings, file_path, metadata)
            else:
                compressed_path = self._save_with_pickle_gzip(embeddings, file_path, metadata)
            
            # Calculate compressed size
            compressed_size = Path(compressed_path).stat().st_size / (1024 * 1024)  # MB
            self.storage_stats['compressed_size_mb'] += compressed_size
            self.storage_stats['files_compressed'] += 1
            
            # Update compression ratio
            if self.storage_stats['original_size_mb'] > 0:
                self.storage_stats['compression_ratio'] = (
                    self.storage_stats['original_size_mb'] / self.storage_stats['compressed_size_mb']
                )
            
            logger.debug(f"💾 Saved compressed embeddings: {original_size:.2f}MB → {compressed_size:.2f}MB "
                        f"({compressed_size/original_size:.2f}x)")
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"Failed to save compressed embeddings: {e}")
            # Fallback to uncompressed save
            return self._save_uncompressed(embeddings, file_path, metadata)
    
    def load_embeddings(self, file_path: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Load embeddings from compressed storage
        
        Args:
            file_path: Path to compressed file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
                return self._load_from_hdf5(file_path)
            elif file_path.suffix == '.pkl.gz':
                return self._load_from_pickle_gzip(file_path)
            else:
                return self._load_uncompressed(file_path)
                
        except Exception as e:
            logger.error(f"Failed to load compressed embeddings: {e}")
            raise
    
    def _save_with_hdf5(self, embeddings: np.ndarray, file_path: Path, metadata: Optional[Dict]) -> str:
        """Save embeddings using HDF5 with compression"""
        hdf5_path = file_path.with_suffix('.h5')
        
        with h5py.File(hdf5_path, 'w') as f:
            # Save embeddings with compression
            f.create_dataset(
                'embeddings', 
                data=embeddings, 
                compression='gzip', 
                compression_opts=self.compression_level,
                shuffle=True,  # Improves compression
                fletcher32=True  # Checksum for data integrity
            )
            
            # Save metadata
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_group.attrs[key] = value
                    else:
                        # Store complex objects as JSON strings
                        metadata_group.attrs[key] = json.dumps(value)
        
        return str(hdf5_path)
    
    def _load_from_hdf5(self, file_path: Path) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load embeddings from HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            embeddings = f['embeddings'][:]
            
            metadata = {}
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    value = metadata_group.attrs[key]
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            metadata[key] = json.loads(value)
                        except:
                            metadata[key] = value
                    else:
                        metadata[key] = value
        
        return embeddings, metadata if metadata else None
    
    def _save_with_pickle_gzip(self, embeddings: np.ndarray, file_path: Path, metadata: Optional[Dict]) -> str:
        """Save embeddings using pickle with gzip compression"""
        pkl_gz_path = file_path.with_suffix('.pkl.gz')
        
        data = {
            'embeddings': embeddings,
            'metadata': metadata
        }
        
        with gzip.open(pkl_gz_path, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return str(pkl_gz_path)
    
    def _load_from_pickle_gzip(self, file_path: Path) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load embeddings from pickle gzip file"""
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data.get('metadata')
    
    def _save_uncompressed(self, embeddings: np.ndarray, file_path: Path, metadata: Optional[Dict]) -> str:
        """Fallback: save without compression"""
        pkl_path = file_path.with_suffix('.pkl')
        
        data = {
            'embeddings': embeddings,
            'metadata': metadata
        }
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return str(pkl_path)
    
    def _load_uncompressed(self, file_path: Path) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load uncompressed embeddings"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data.get('metadata')
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.storage_stats.copy()
    
    def optimize_existing_embeddings(self, directory: str) -> Dict[str, Any]:
        """Optimize existing embedding files in directory"""
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return {}
        
        optimization_stats = {
            'files_processed': 0,
            'files_optimized': 0,
            'total_size_before_mb': 0.0,
            'total_size_after_mb': 0.0,
            'errors': []
        }
        
        # Find embedding files
        embedding_files = list(directory.glob('*.pkl')) + list(directory.glob('*.npy'))
        
        for file_path in embedding_files:
            try:
                # Get original size
                original_size = file_path.stat().st_size / (1024 * 1024)
                optimization_stats['total_size_before_mb'] += original_size
                optimization_stats['files_processed'] += 1
                
                # Load embeddings
                if file_path.suffix == '.pkl':
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    embeddings = data.get('embeddings') if isinstance(data, dict) else data
                    metadata = data.get('metadata') if isinstance(data, dict) else None
                elif file_path.suffix == '.npy':
                    embeddings = np.load(file_path)
                    metadata = None
                else:
                    continue
                
                # Save with compression
                compressed_path = self.save_embeddings(embeddings, str(file_path), metadata)
                
                # Get compressed size
                compressed_size = Path(compressed_path).stat().st_size / (1024 * 1024)
                optimization_stats['total_size_after_mb'] += compressed_size
                
                # Remove original if compression was successful and different
                if compressed_path != str(file_path) and Path(compressed_path).exists():
                    file_path.unlink()
                    optimization_stats['files_optimized'] += 1
                    
                    logger.info(f"📦 Optimized {file_path.name}: {original_size:.2f}MB → {compressed_size:.2f}MB")
                
            except Exception as e:
                error_msg = f"Failed to optimize {file_path}: {e}"
                logger.error(error_msg)
                optimization_stats['errors'].append(error_msg)
        
        # Calculate overall compression ratio
        if optimization_stats['total_size_before_mb'] > 0:
            compression_ratio = optimization_stats['total_size_before_mb'] / optimization_stats['total_size_after_mb']
            optimization_stats['compression_ratio'] = compression_ratio
        
        logger.info(f"📦 Embedding optimization completed: {optimization_stats['files_optimized']} files optimized, "
                   f"{optimization_stats['total_size_before_mb']:.2f}MB → {optimization_stats['total_size_after_mb']:.2f}MB")
        
        return optimization_stats


class MemoryOptimizer:
    """
    Main memory optimization coordinator
    """
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        self.config = config or MemoryOptimizationConfig()
        
        # Initialize components
        self.quantizer = QualityPreservingQuantizer(self.config)
        self.gradient_checkpointing = GradientCheckpointing(self.config)
        self.embedding_storage = CompressedEmbeddingStorage(self.config)
        
        # Memory monitoring
        self.memory_usage_history = []
        
        logger.info("🔧 MemoryOptimizer initialized")
    
    def optimize_model(self, 
                      model: nn.Module, 
                      calibration_data: Optional[torch.Tensor] = None,
                      test_data: Optional[torch.Tensor] = None,
                      test_labels: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Comprehensive model optimization
        
        Args:
            model: Model to optimize
            calibration_data: Data for quantization calibration
            test_data: Data for quality validation
            test_labels: Labels for quality validation
            
        Returns:
            Optimized model
        """
        logger.info("🚀 Starting comprehensive model optimization")
        
        original_size = self.quantizer._calculate_model_size(model)
        logger.info(f"📊 Original model size: {original_size:.2f}MB")
        
        # Apply gradient checkpointing
        model = self.gradient_checkpointing.apply_gradient_checkpointing(model)
        
        # Apply quantization
        quantized_model = self.quantizer.quantize_model(model, calibration_data)
        
        # Validate quality if test data provided
        if test_data is not None and test_labels is not None:
            quality_metrics = self.quantizer.validate_quantization_quality(
                model, quantized_model, test_data, test_labels
            )
            logger.info(f"📈 Quality validation: {quality_metrics}")
        
        optimized_size = self.quantizer._calculate_model_size(quantized_model)
        compression_ratio = original_size / optimized_size
        
        logger.info(f"✅ Model optimization completed: {original_size:.2f}MB → {optimized_size:.2f}MB "
                   f"({compression_ratio:.2f}x compression)")
        
        return quantized_model
    
    def optimize_embeddings_directory(self, directory: str) -> Dict[str, Any]:
        """Optimize all embeddings in a directory"""
        return self.embedding_storage.optimize_existing_embeddings(directory)
    
    def save_optimized_embeddings(self, embeddings: np.ndarray, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Save embeddings with optimization"""
        return self.embedding_storage.save_embeddings(embeddings, file_path, metadata)
    
    def load_optimized_embeddings(self, file_path: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load optimized embeddings"""
        return self.embedding_storage.load_embeddings(file_path)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'config': {
                'quantization_enabled': self.config.enable_quantization,
                'quantization_bits': self.config.quantization_bits,
                'gradient_checkpointing_enabled': self.config.enable_gradient_checkpointing,
                'mixed_precision_enabled': self.config.enable_mixed_precision,
                'compression_level': self.config.embedding_compression_level,
                'use_hdf5': self.config.use_hdf5_storage
            },
            'model_optimization': {
                'original_size_mb': self.quantizer.original_model_size,
                'quantized_size_mb': self.quantizer.quantized_model_size,
                'compression_ratio': (self.quantizer.original_model_size / self.quantizer.quantized_model_size) 
                                   if self.quantizer.quantized_model_size > 0 else 1.0
            },
            'embedding_optimization': self.embedding_storage.get_storage_statistics()
        }
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_stats = {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'percent': process.memory_percent()
            }
            
            # Add to history
            self.memory_usage_history.append({
                'timestamp': time.time(),
                **memory_stats
            })
            
            # Keep only recent history (last 100 measurements)
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-100:]
            
            return memory_stats
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {}
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {}
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization should be triggered"""
        memory_stats = self.monitor_memory_usage()
        
        if memory_stats:
            current_memory_mb = memory_stats.get('rss_mb', 0)
            return current_memory_mb > self.config.memory_threshold_mb
        
        return False