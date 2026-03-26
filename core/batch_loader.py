"""
Batch Image Loader with Optimization
Efficiently loads and processes image batches to minimize I/O overhead.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Iterator
from pathlib import Path


class BatchImageLoader:
    """
    Loads image batches efficiently from disk.
    Supports prefetching and memory efficiency for large-scale processing.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self, image_folder: str, batch_size: int = 8):
        """
        Initialize batch loader.
        
        Args:
            image_folder: Path to folder containing images
            batch_size: Number of images to load per batch
        """
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_files = self._discover_images()
        self.total_batches = (len(self.image_files) + batch_size - 1) // batch_size
    
    def _discover_images(self) -> List[str]:
        """Discover all supported image files in folder."""
        image_files = []
        
        for filename in os.listdir(self.image_folder):
            ext = Path(filename).suffix.lower()
            if ext in self.SUPPORTED_FORMATS:
                image_files.append(filename)
        
        return sorted(image_files)
    
    def get_file_list(self) -> List[str]:
        """Get list of all discovered image files."""
        return self.image_files.copy()
    
    def get_total_batches(self) -> int:
        """Get total number of batches."""
        return self.total_batches
    
    def load_batch(self, filenames: List[str]) -> List[Tuple[str, np.ndarray, bool]]:
        """
        Load a batch of images from disk.
        
        Args:
            filenames: List of filenames to load
            
        Returns:
            List of tuples: (filename, image_array, success)
            Image arrays are in BGR format (OpenCV convention)
        """
        batch = []
        
        for filename in filenames:
            img_path = os.path.join(self.image_folder, filename)
            
            if not os.path.exists(img_path):
                batch.append((filename, None, False))
                continue
            
            try:
                image = cv2.imread(img_path)
                if image is None:
                    batch.append((filename, None, False))
                else:
                    batch.append((filename, image, True))
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                batch.append((filename, None, False))
        
        return batch
    
    def load_batch_filtered(self, filenames: List[str], min_width: int = 100, 
                           min_height: int = 100) -> List[Tuple[str, np.ndarray, bool]]:
        """
        Load batch with size filtering.
        
        Args:
            filenames: List of filenames to load
            min_width: Minimum image width
            min_height: Minimum image height
            
        Returns:
            Same format as load_batch, but success=False if image is too small
        """
        batch = self.load_batch(filenames)
        filtered = []
        
        for filename, image, success in batch:
            if success and image is not None:
                h, w = image.shape[:2]
                if w >= min_width and h >= min_height:
                    filtered.append((filename, image, True))
                else:
                    filtered.append((filename, image, False))
            else:
                filtered.append((filename, image, success))
        
        return filtered
    
    def iterate_batches(self, filter_list: List[str] = None) -> Iterator:
        """
        Iterate over image batches.
        
        Args:
            filter_list: If provided, only load images in this list
            
        Yields:
            Tuples of (batch_index, batch_filenames, batch_images_dict)
        """
        files_to_process = filter_list if filter_list else self.image_files
        
        for batch_idx in range(0, len(files_to_process), self.batch_size):
            batch_files = files_to_process[batch_idx:batch_idx + self.batch_size]
            batch_data = self.load_batch(batch_files)
            
            # Return as dict for easier processing
            batch_dict = {
                filename: {
                    'image': image,
                    'success': success
                }
                for filename, image, success in batch_data
            }
            
            yield {
                'batch_index': batch_idx // self.batch_size,
                'batch_size': len(batch_files),
                'files': batch_files,
                'data': batch_dict
            }


class ParallelBatchLoader:
    """
    Multi-threaded batch image loader (for I/O-only operations, not model inference).
    """
    
    def __init__(self, image_folder: str, batch_size: int = 8, num_workers: int = 2):
        """
        Initialize parallel batch loader.
        
        Args:
            image_folder: Path to folder containing images
            batch_size: Number of images per batch
            num_workers: Number of loader threads (for I/O parallelism)
        """
        self.loader = BatchImageLoader(image_folder, batch_size)
        self.num_workers = num_workers
    
    def get_file_list(self) -> List[str]:
        """Get list of all image files."""
        return self.loader.get_file_list()
    
    def get_total_batches(self) -> int:
        """Get total number of batches."""
        return self.loader.get_total_batches()
    
    def iterate_batches(self) -> Iterator:
        """
        Iterate over batches using standard loader
        (Real parallelization would use ThreadPoolExecutor, but ONNX inference
        must remain sequential on same GPU/model session).
        """
        return self.loader.iterate_batches()


# Utility function for filtering
def filter_already_processed(all_files: List[str], processed_files: set) -> List[str]:
    """
    Filter out already-processed files.
    
    Args:
        all_files: List of all available image files
        processed_files: Set of filenames already processed
        
    Returns:
        List of files not yet processed
    """
    return [f for f in all_files if f not in processed_files]


if __name__ == '__main__':
    # Example usage
    loader = BatchImageLoader('./data/images', batch_size=8)
    
    print(f"Total images: {len(loader.get_file_list())}")
    print(f"Total batches: {loader.get_total_batches()}")
    
    # Iterate over batches
    for batch_info in loader.iterate_batches():
        print(f"Batch {batch_info['batch_index']}: {batch_info['batch_size']} files")
        for filename in batch_info['files']:
            success = batch_info['data'][filename]['success']
            print(f"  ✓" if success else f"  ✗", f"  {filename}")
