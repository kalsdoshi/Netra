"""
GPU Detection and Optimization Utilities
Detects available GPU resources and provides recommendations for optimal configuration.
"""

import os
import numpy as np
from typing import Dict, Tuple


class GPUDetector:
    """Detect GPU availability and provide optimization recommendations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.gpu_available = False
        self.device_id = -1
        self.gpu_memory_mb = 0
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect CUDA/GPU availability and properties."""
        try:
            # Check ONNX Runtime providers
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    self.gpu_available = True
                    self.device_id = 0
                    print("✅ CUDA GPU detected for ONNX Runtime")
                else:
                    print("ℹ️  No CUDA provider available in ONNX Runtime")
            except ImportError:
                pass
            
            # Check PyTorch/CUDA (fallback)
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_available = True
                    self.device_id = 0
                    try:
                        self.gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                        print(f"✅ PyTorch CUDA GPU detected ({self.gpu_memory_mb:.0f} MB)")
                    except:
                        print("✅ PyTorch CUDA GPU available")
            except ImportError:
                pass
            
        except Exception as e:
            print(f"⚠️  GPU detection error: {e}")
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available
    
    def get_device_id(self) -> int:
        """
        Get device ID for GPU usage.
        Returns: 0 for GPU, -1 for CPU
        """
        return self.device_id if self.gpu_available else -1
    
    def get_recommendations(self) -> Dict:
        """
        Get optimization recommendations based on available hardware.
        Returns dict with batch_size, num_workers, use_gpu flags.
        """
        if self.gpu_available:
            return {
                'use_gpu': True,
                'batch_size': 32,  # GPU can handle larger batches
                'embedding_batch_size': 64,
                'gpu_memory_mb': self.gpu_memory_mb,
                'device_id': self.device_id,
                'note': 'GPU accelerated - optimal for speed'
            }
        else:
            # CPU-only: smaller batches, use ONNX internal parallelism
            num_cpus = os.cpu_count() or 4
            return {
                'use_gpu': False,
                'batch_size': 8,  # CPU processes smaller batches
                'embedding_batch_size': 16,
                'gpu_memory_mb': 0,
                'device_id': -1,
                'num_cpus': num_cpus,
                'note': 'CPU mode - ONNX Runtime uses internal parallelism'
            }
    
    def estimate_memory_usage(self, num_faces: int, embedding_dim: int = 512) -> Dict:
        """
        Estimate memory usage for storing embeddings.
        Args:
            num_faces: Number of face embeddings
            embedding_dim: Dimension of embeddings (default 512)
        Returns dict with memory estimates in MB
        """
        embedding_bytes = num_faces * embedding_dim * 4  # float32 = 4 bytes
        embedding_mb = embedding_bytes / (1024**2)
        
        # FAISS index overhead (rough estimate)
        faiss_overhead_mb = embedding_mb * 0.2  # ~20% overhead for indexing
        
        # Metadata storage (rough estimate: ~500 bytes per face)
        metadata_mb = (num_faces * 500) / (1024**2)
        
        total_mb = embedding_mb + faiss_overhead_mb + metadata_mb
        
        return {
            'embeddings_mb': embedding_mb,
            'faiss_index_overhead_mb': faiss_overhead_mb,
            'metadata_mb': metadata_mb,
            'total_estimated_mb': total_mb,
            'available_gpu_mb': self.gpu_memory_mb if self.gpu_available else 0,
            'fits_in_gpu': (total_mb < self.gpu_memory_mb) if self.gpu_available else False
        }


def get_processing_config(target_image_count: int) -> Dict:
    """
    Get recommended processing configuration based on available hardware and data size.
    
    Args:
        target_image_count: Approximate number of images to process
        
    Returns:
        Dictionary with optimal configuration parameters
    """
    gpu = GPUDetector()
    recommendations = gpu.get_recommendations()
    
    # Memory estimate (assume ~3 faces per image on average)
    estimated_faces = target_image_count * 3
    memory_usage = gpu.estimate_memory_usage(estimated_faces)
    
    config = {
        **recommendations,
        'target_image_count': target_image_count,
        'estimated_faces': estimated_faces,
        'memory_usage': memory_usage,
        'image_read_batch_size': 8,  # Read 8 images at a time from disk
        'detector_batch_capable': True,  # InsightFace can handle batches
    }
    
    # Adjust if memory constrained
    if not gpu.gpu_available and memory_usage['total_estimated_mb'] > 4096:  # > 4GB
        config['image_read_batch_size'] = 4
        config['embedding_batch_size'] = 8
    
    return config


def print_system_info():
    """Print system information and GPU status."""
    gpu = GPUDetector()
    print("\n" + "="*50)
    print("🖥️  SYSTEM INFORMATION")
    print("="*50)
    print(f"GPU Available: {'✅ Yes' if gpu.is_available() else '❌ No'}")
    print(f"GPU Device ID: {gpu.get_device_id()}")
    if gpu.gpu_available:
        print(f"GPU Memory: {gpu.gpu_memory_mb:.0f} MB")
    print(f"CPU Cores: {os.cpu_count()}")
    
    recs = gpu.get_recommendations()
    print(f"\nRecommendations:")
    print(f"  Use GPU: {recs['use_gpu']}")
    print(f"  Batch Size: {recs['batch_size']}")
    print(f"  Embedding Batch Size: {recs['embedding_batch_size']}")
    print(f"  Note: {recs['note']}")
    print("="*50 + "\n")


if __name__ == '__main__':
    print_system_info()
    
    # Test for 50K image set
    config = get_processing_config(50000)
    print(f"Configuration for 50K images:")
    print(f"  Estimated faces: {config['estimated_faces']}")
    print(f"  Total memory needed: {config['memory_usage']['total_estimated_mb']:.0f} MB")
    print(f"  Fits in GPU: {config['memory_usage']['fits_in_gpu']}")
