import os
from pathlib import Path
import json
from datetime import datetime
from models.face_analyzer import LightFaceAnalyzer
from models.aesthetic_analyzer import AestheticAnalyzer
from utils.image_quality import ImageQualityAnalyzer
from utils.visualization import ImageVisualizer
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

class ImageDeduplicator:
    def __init__(self, folder_path):
        """Initialize image deduplicator"""
        self.folder_path = Path(folder_path)
        self.image_hashes = {}
        self.similar_groups = []
        self.duplicates = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Initialize analyzers
        print(f"\nInitializing analyzers...")
        self.quality_analyzer = ImageQualityAnalyzer()
        self.visualizer = ImageVisualizer()
        
        # Cache
        self.cache = {}

    def get_image_quality_score(self, image_path):
        """Get image quality score"""
        score, metrics = self.quality_analyzer.analyze_image(image_path)
        return score, metrics

    def preprocess_image(self, path):
        """Preprocess image and cache result"""
        if path in self.cache:
            return self.cache[path]
            
        try:
            # Read image and downsample
            img = cv2.imread(str(path))
            if img is None:
                return None
                
            # Downsample to fixed size
            img = cv2.resize(img, (128, 128))  # Just a little smaller
            
            # Compute hash
            img_hash = self.quality_analyzer.compute_hash(path)
            
            # Compute histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], 
                              [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            result = {
                'image': img,
                'hash': img_hash,
                'hist': hist,
                'gray': gray
            }
            
            self.cache[path] = result
            return result
            
        except Exception as e:
            print(f"Error preprocessing {path}: {str(e)}")
            return None

    def compute_image_similarity(self, path1, path2):
        """Compute similarity between two images"""
        # Get preprocessed results
        data1 = self.preprocess_image(path1)
        data2 = self.preprocess_image(path2)
        
        if not data1 or not data2:
            return 0
            
        # 1. Scene hash comparison (detect background similarity)
        hash_diff = sum(c1 != c2 for c1, c2 in zip(data1['hash'], data2['hash']))
        hash_similarity = 1 - (hash_diff / 64.0)
        
        # If scenes are completely different, return immediately
        if hash_similarity < 0.3:  # Adjust back to 0.3
            return 0
            
        # 2. Histogram comparison (detect overall color distribution)
        hist_similarity = cv2.compareHist(data1['hist'], data2['hist'], 
                                        cv2.HISTCMP_CORREL)
        
        # 3. SSIM comparison (detect structural similarity)
        ssim_score, _ = ssim(data1['gray'], data2['gray'], full=True)
        
        # 4. Local area comparison (detect detail changes)
        img1 = data1['image']
        img2 = data2['image']
        
        # Split image into 3x3 grid
        h, w = img1.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        
        local_similarities = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                region1 = img1[y1:y2, x1:x2]
                region2 = img2[y1:y2, x1:x2]
                
                # Compute similarity of local areas
                region_hist1 = cv2.calcHist([region1], [0, 1, 2], None, [8, 8, 8], 
                                          [0, 256, 0, 256, 0, 256]).flatten()
                region_hist2 = cv2.calcHist([region2], [0, 1, 2], None, [8, 8, 8], 
                                          [0, 256, 0, 256, 0, 256]).flatten()
                
                region_similarity = cv2.compareHist(region_hist1, region_hist2, 
                                                  cv2.HISTCMP_CORREL)
                local_similarities.append(region_similarity)
        
        # Compute average similarity of local areas
        local_mean = np.mean(local_similarities)
        
        # Compute final similarity
        weights = {
            'hash': 0.35,       # Keep hash weight
            'histogram': 0.25,  # Keep histogram weight
            'ssim': 0.3,       # Keep SSIM weight
            'local': 0.1       # Keep local comparison weight
        }
        
        similarity = (
            weights['hash'] * hash_similarity +
            weights['histogram'] * max(0, hist_similarity) +
            weights['ssim'] * max(0, ssim_score) +
            weights['local'] * local_mean
        )
        
        return similarity

    def find_similar_images(self, similarity_threshold=0.65):
        """Find similar images"""
        image_files = [
            f for f in self.folder_path.rglob("*")
            if f.suffix.lower() in self.supported_formats
        ]
        
        total = len(image_files)
        print(f"Found {total} images to process...")
        
        # Preprocess all images
        print("Preprocessing images...")
        for i, path in enumerate(image_files):
            print(f"\rPreprocessing: {i+1}/{total}", end="")
            self.preprocess_image(str(path))
        print("\nPreprocessing complete!")
        
        # Compare images
        processed = set()
        similar_groups = []
        self.duplicates = []  # Clear old duplicates list
        
        print("\nComparing images...")
        for i, path1 in enumerate(image_files):
            if str(path1) in processed:
                continue
            
            print(f"\rComparing images: {i+1}/{total}...", end="")
            
            # Find all images similar to the current image
            current_group = {str(path1)}  # Use set to store similar images
            
            for path2 in image_files:
                if str(path2) not in processed and str(path2) != str(path1):
                    similarity = self.compute_image_similarity(str(path1), str(path2))
                    if similarity > similarity_threshold:
                        current_group.add(str(path2))
                        print(f"\nFound similar: {path1} <-> {path2} (similarity: {similarity:.3f})")
            
            # If similar images are found
            if len(current_group) > 1:
                group_images = []
                print(f"\nProcessing group with {len(current_group)} images:")
                for path in current_group:
                    score, metrics = self.get_image_quality_score(path)
                    group_images.append((path, score, metrics))
                    processed.add(path)
                    print(f"  - {path} (score: {score:.1f})")
                
                # Save to image_hashes and similar_groups
                self.image_hashes[str(path1)] = group_images
                similar_groups.append(group_images)
                
                # Add to duplicates list (excluding highest quality image)
                sorted_images = sorted(group_images, key=lambda x: x[1], reverse=True)
                self.duplicates.extend([img[0] for img in sorted_images[1:]])
                
                print(f"Added group {len(similar_groups)} with {len(group_images)} images")
        
        print("\nImage comparison complete!")
        print(f"Total groups found: {len(similar_groups)}")
        for i, group in enumerate(similar_groups, 1):
            print(f"\nGroup {i} ({len(group)} images):")
            for path, score, _ in group:
                print(f"  - {path} (score: {score:.1f})")
        
        # Show all similar image groups
        if similar_groups:
            print(f"\nFound {len(similar_groups)} groups of similar images.")
            print("Use left/right arrow keys to navigate between groups.")
            self.visualizer.show_all_groups(similar_groups)
            
            # Save duplicates file
            output_file = self.save_duplicate_list()
            print(f"\nFound {len(self.duplicates)} duplicate images.")
            print(f"Duplicate list saved to: {output_file}")
        else:
            print("No duplicate images found.")

    def save_duplicate_list(self):
        """Save duplicate image list"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.folder_path / f"duplicates_{timestamp}.json"
        
        # Reorganize the data to group similar images
        grouped_duplicates = []
        for img_hash, images in self.image_hashes.items():
            if len(images) > 1:
                sorted_images = sorted(images, key=lambda x: x[1], reverse=True)
                group = {
                    "best_quality_image": {
                        "path": sorted_images[0][0],
                        "quality_score": sorted_images[0][1]
                    },
                    "duplicates": [
                        {
                            "path": img[0],
                            "quality_score": img[1]
                        } for img in sorted_images[1:]
                    ]
                }
                grouped_duplicates.append(group)
        
        duplicate_info = {
            "summary": {
                "timestamp": timestamp,
                "total_groups": len(grouped_duplicates),
                "total_duplicates": sum(len(group["duplicates"]) for group in grouped_duplicates)
            },
            "similar_image_groups": grouped_duplicates
        }
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(duplicate_info, f, indent=4, ensure_ascii=False)
        
        return output_file

    def get_adaptive_weights(self, image_metrics):
        """Adjust weights based on image content"""
        weights = {
            'technical': 0.3,
            'face': 0.3,
            'aesthetic': 0.2,
            'scene': 0.2
        }
        
        # Adjust based on scene type
        if image_metrics['scene_type'] == 'portrait':
            weights['face'] = 0.4
            weights['technical'] = 0.2
        elif image_metrics['scene_type'] == 'landscape':
            weights['aesthetic'] = 0.3
            weights['face'] = 0.1
        
        return weights

    # ... (rest of the ImageDeduplicator class methods) 