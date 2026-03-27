import os
import cv2
import numpy as np
from tqdm import tqdm

from core.cluster import HDBSCANCluster
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.visualize import save_clusters
from storage.store import Storage
from core.faiss_index import FaissIndex
from core.config import Config
from core.object_detector import ObjectDetector
from core.gpu_utils import GPUDetector, get_processing_config, print_system_info
from core.batch_loader import BatchImageLoader, filter_already_processed
from core.performance import PerformanceMonitor, get_monitor


# ─── Face Quality Gate ─────────────────────────────────────────
MIN_FACE_SIZE = 40       # pixels — skip faces smaller than this
MIN_LAPLACIAN_VAR = 15.0 # blur threshold — skip blurry face crops


def _face_quality_ok(image, bbox):
    """
    Check if a face crop meets minimum quality requirements.
    Returns False for faces that are too small or too blurry,
    which would degrade embedding quality and clustering accuracy.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    # Skip tiny faces — unreliable embeddings
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    
    # Skip blurry faces — check Laplacian variance
    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < MIN_LAPLACIAN_VAR:
        return False
    
    return True


class ImageProcessor:
    def __init__(self, image_folder, use_gpu=None):
        """
        Initialize ImageProcessor with optional GPU acceleration.
        
        Args:
            image_folder: Path to folder containing images
            use_gpu: If None, auto-detect GPU. If True/False, force it.
        """
        self.image_folder = image_folder
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            gpu_detector = GPUDetector()
            use_gpu = gpu_detector.is_available()
        
        self.detector = FaceDetector(use_gpu=use_gpu)
        self.embedder = FaceEmbedder()
        
        self.embeddings = []
        self.metadata = []

        self.storage = Storage()
        self.object_detector = ObjectDetector(min_confidence=0.4)
        
        # Performance monitoring
        self.monitor = get_monitor()
        
        # Batch loader for efficient I/O
        self.batch_loader = BatchImageLoader(image_folder, batch_size=8)

    def _process_single_image(self, img_name, image):
        """
        Process a single image: detect faces + objects, extract embeddings.
        Note: Image is passed as parameter to allow batch loading.
        """
        try:
            if image is None:
                return [], []

            faces = self.detector.detect(image)
            objects = self.object_detector.detect(image)
            
            local_embeddings = []
            local_metadata = []

            for idx, f in enumerate(faces):
                bbox = f["bbox"]
                h, w, _ = image.shape
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Face quality gate — skip tiny/blurry faces
                if not _face_quality_ok(image, [x1, y1, x2, y2]):
                    continue

                face_crop = image[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                embedding = self.embedder.get_embedding(f["face"])
                local_embeddings.append(embedding)
                
                local_metadata.append({
                    "image": img_name,
                    "face_id": int(idx),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "objects": objects,
                })
            return local_embeddings, local_metadata
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            return [], []


    def process(self):
        """
        Optimized main processing pipeline with batch loading and performance monitoring.
        """
        print_system_info()
        
        # Load previous data
        old_embeddings = self.storage.load_embeddings()
        old_metadata = self.storage.load_metadata()

        processed_images = set()

        if old_metadata:
            processed_images = set([m["image"] for m in old_metadata])
            print(f"📂 Found {len(processed_images)} already processed images")
        
        # Get all available images
        all_images = self.batch_loader.get_file_list()
        images_to_process = filter_already_processed(all_images, processed_images)
        new_embeddings = np.empty((0, 512), dtype=np.float32)
        
        if not images_to_process:
            print("⚡ No new images found. Using existing data.")
            embeddings_array = old_embeddings
            metadata = old_metadata
        else:
            print(f"⚡ Processing {len(images_to_process)} new images using batch loading...")
            
            new_embeddings_list = []
            new_metadata_list = []

            # Batch processing with performance monitoring
            self.monitor.start_operation('batch_processing')
            
            batch_count = 0
            for batch_idx in range(0, len(images_to_process), self.batch_loader.batch_size):
                batch_files = images_to_process[batch_idx:batch_idx + self.batch_loader.batch_size]
                
                # Load batch from disk
                self.monitor.start_operation('batch_load')
                batch_data = self.batch_loader.load_batch(batch_files)
                self.monitor.end_operation('batch_load', count=len(batch_files))
                
                # Process each image in batch
                self.monitor.start_operation('feature_extraction')
                for filename, image, success in batch_data:
                    if not success:
                        continue
                    
                    embs, metas = self._process_single_image(filename, image)
                    if embs:
                        new_embeddings_list.extend(embs)
                        new_metadata_list.extend(metas)
                
                self.monitor.end_operation('feature_extraction', count=len(batch_files))
                batch_count += 1
                
                # Progress
                processed_count = min(batch_idx + self.batch_loader.batch_size, len(images_to_process))
                print(f"  Processed {processed_count}/{len(images_to_process)} images, "
                      f"extracted {len(new_embeddings_list)} faces")
            
            self.monitor.end_operation('batch_processing', count=len(images_to_process))
            
            self.embeddings.extend(new_embeddings_list)
            self.metadata.extend(new_metadata_list)

            # --- PREPARE NEW EMBEDDINGS ---
            new_embeddings = (
                np.array(new_embeddings_list, dtype=np.float32)
                if new_embeddings_list
                else np.empty((0, 512), dtype=np.float32)
            )

            # --- MERGE LOGIC ---
            # CASE 1: First run (no old data)
            if old_embeddings is None:
                embeddings_array = new_embeddings
                metadata = self.metadata

            # CASE 2: New images had no valid faces
            elif new_embeddings.shape[0] == 0:
                print("⚡ No valid new faces extracted. Reusing existing embeddings and metadata.")
                embeddings_array = old_embeddings
                metadata = old_metadata if old_metadata is not None else []

            # CASE 3: Normal merge (old + new)
            else:
                embeddings_array = np.vstack([old_embeddings, new_embeddings])
                if old_metadata is None:
                    old_metadata = []
                metadata = old_metadata + self.metadata

        # --- FAISS INDEX ---
        print("⚡ Processing FAISS Index...")
        self.monitor.start_operation('faiss_indexing')
        
        faiss_index = FaissIndex(dim=512)
        old_index = self.storage.load_faiss_index()

        # Smart incremental update decision
        if old_index is not None and old_embeddings is not None and new_embeddings.shape[0] == 0:
            print("⚡ Using existing FAISS index (no new embeddings)")
            faiss_index.index = old_index
            faiss_index.index_size = old_index.ntotal if hasattr(old_index, "ntotal") else len(old_embeddings)
        elif old_index is not None and old_embeddings is not None and new_embeddings.shape[0] > 0:
            print("⚡ Using incremental FAISS index update")
            faiss_index.index = old_index
            faiss_index.index_size = old_index.ntotal if hasattr(old_index, "ntotal") else len(old_embeddings)
            update_stats = faiss_index.add_incremental(new_embeddings)
            print(f"  Updated: {update_stats['added']} new embeddings, "
                  f"Total size: {update_stats['new_size']}, "
                  f"Speed: {update_stats['speed_embeddings_per_sec']:.1f} emb/sec")
        elif embeddings_array is not None and embeddings_array.size > 0:
            print("⚡ Building FAISS index from scratch")
            faiss_index.build(embeddings_array)

        self.monitor.end_operation('faiss_indexing')
        self.storage.save_faiss_index(faiss_index.index)    

        # --- CLUSTERING LOGIC ---
        print("⚡ Processing Clustering...")
        self.monitor.start_operation('clustering')
        
        config = Config()
        eps = config.get("clustering", "eps")
        min_samples = config.get("clustering", "min_samples")

        old_clusters = self.storage.load_clusters()

        # CASE 1: No new data → reuse
        if new_embeddings.shape[0] == 0 and old_clusters is not None:
            print("⚡ Using existing clusters (no recomputation)")
            cluster_dict = old_clusters

        # CASE 2: Incremental (keep your existing logic)
        elif old_clusters is not None and old_embeddings is not None:
            print("⚡ Incremental clustering (adding new faces)")

            from core.cluster import assign_to_clusters

            cluster_dict = assign_to_clusters(
                embeddings_array,
                new_embeddings,
                old_clusters,
                len(old_embeddings),
                threshold=0.5
            )

        # CASE 3: First run → HDBSCAN
        else:
            print("⚡ First-time clustering (HDBSCAN)")

            clusterer = HDBSCANCluster(
                min_cluster_size=min_samples,
                min_samples=1,
                cluster_selection_epsilon=0.35
            )
            clusters = clusterer.cluster(embeddings_array)

            cluster_dict = {
                f"person_{i}": cluster
                for i, cluster in enumerate(clusters)
            }
        
        self.monitor.end_operation('clustering')

        # --- OUTPUT ---
        print(f"Total clusters (people): {len(cluster_dict)}")

        # --- GRAPH BUILDING ---
        print("⚡ Building Relational Graph...")
        self.monitor.start_operation('graph_building')
        
        from core.graph_builder import GraphBuilder
        graph_builder = GraphBuilder(embeddings_array, metadata, cluster_dict, threshold=0.15)
        graph_data = graph_builder.build_graph()
        
        self.monitor.end_operation('graph_building')

        # --- VISUALIZATION ---
        print("⚡ Saving cluster visualizations...")
        save_clusters(cluster_dict, metadata)

        # --- SAVE (VERY IMPORTANT ORDER) ---
        self.storage.save_embeddings(embeddings_array)
        self.storage.save_metadata(metadata)
        self.storage.save_clusters(cluster_dict)
        self.storage.save_graph(graph_data)

        # --- PERFORMANCE REPORT ---
        print("\n")
        self.monitor.print_report()
        
        bottlenecks = self.monitor.get_bottleneck_analysis()
        print("Bottleneck Analysis (slowest operations):")
        for i, b in enumerate(bottlenecks[:3], 1):
            print(f"  {i}. {b['operation']}: {b['percentage']:.1f}% ({b['total_time']:.2f}s)")

        return embeddings_array, metadata, cluster_dict
