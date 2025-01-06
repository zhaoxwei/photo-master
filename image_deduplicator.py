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
        """初始化图片去重器"""
        self.folder_path = Path(folder_path)
        self.image_hashes = {}
        self.similar_groups = []
        self.duplicates = []
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 初始化分析器
        print(f"\nInitializing analyzers...")
        self.quality_analyzer = ImageQualityAnalyzer()
        self.visualizer = ImageVisualizer()
        
        # 缓存
        self.cache = {}

    def get_image_quality_score(self, image_path):
        """获取图片质量分数"""
        score, metrics = self.quality_analyzer.analyze_image(image_path)
        return score, metrics

    def preprocess_image(self, path):
        """预处理图片并缓存结果"""
        if path in self.cache:
            return self.cache[path]
            
        try:
            # 读取图片并降采样
            img = cv2.imread(str(path))
            if img is None:
                return None
                
            # 降采样到固定大小
            img = cv2.resize(img, (128, 128))  # 只改小一点点
            
            # 计算哈希值
            img_hash = self.quality_analyzer.compute_hash(path)
            
            # 计算直方图
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], 
                              [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # 转换为灰度图
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
        """计算两张图片的相似度"""
        # 获取预处理结果
        data1 = self.preprocess_image(path1)
        data2 = self.preprocess_image(path2)
        
        if not data1 or not data2:
            return 0
            
        # 1. 场景哈希比较 (检测背景是否相似)
        hash_diff = sum(c1 != c2 for c1, c2 in zip(data1['hash'], data2['hash']))
        hash_similarity = 1 - (hash_diff / 64.0)
        
        # 如果场景完全不同，直接返回
        if hash_similarity < 0.3:  # 调整回0.3
            return 0
            
        # 2. 直方图比较 (检测整体颜色分布)
        hist_similarity = cv2.compareHist(data1['hist'], data2['hist'], 
                                        cv2.HISTCMP_CORREL)
        
        # 3. SSIM比较 (检测结构相似度)
        ssim_score, _ = ssim(data1['gray'], data2['gray'], full=True)
        
        # 4. 局部区域比较 (检测细节变化)
        img1 = data1['image']
        img2 = data2['image']
        
        # 将图像分成3x3的网格
        h, w = img1.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        
        local_similarities = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                region1 = img1[y1:y2, x1:x2]
                region2 = img2[y1:y2, x1:x2]
                
                # 计算局部区域的相似度
                region_hist1 = cv2.calcHist([region1], [0, 1, 2], None, [8, 8, 8], 
                                          [0, 256, 0, 256, 0, 256]).flatten()
                region_hist2 = cv2.calcHist([region2], [0, 1, 2], None, [8, 8, 8], 
                                          [0, 256, 0, 256, 0, 256]).flatten()
                
                region_similarity = cv2.compareHist(region_hist1, region_hist2, 
                                                  cv2.HISTCMP_CORREL)
                local_similarities.append(region_similarity)
        
        # 计算局部区域相似度的平均值
        local_mean = np.mean(local_similarities)
        
        # 计算最终相似度
        weights = {
            'hash': 0.35,       # 保持哈希权重
            'histogram': 0.25,  # 保持直方图权重
            'ssim': 0.3,       # 保持结构相似度权重
            'local': 0.1       # 保持局部比较权重
        }
        
        similarity = (
            weights['hash'] * hash_similarity +
            weights['histogram'] * max(0, hist_similarity) +
            weights['ssim'] * max(0, ssim_score) +
            weights['local'] * local_mean
        )
        
        return similarity

    def find_similar_images(self, similarity_threshold=0.65):
        """查找相似图片"""
        image_files = [
            f for f in self.folder_path.rglob("*")
            if f.suffix.lower() in self.supported_formats
        ]
        
        total = len(image_files)
        print(f"Found {total} images to process...")
        
        # 预处理所有图片
        print("Preprocessing images...")
        for i, path in enumerate(image_files):
            print(f"\rPreprocessing: {i+1}/{total}", end="")
            self.preprocess_image(str(path))
        print("\nPreprocessing complete!")
        
        # 比较图片
        processed = set()
        similar_groups = []
        self.duplicates = []  # 清空旧的duplicates列表
        
        print("\nComparing images...")
        for i, path1 in enumerate(image_files):
            if str(path1) in processed:
                continue
            
            print(f"\rComparing images: {i+1}/{total}...", end="")
            
            # 找出所有与当前图片相似的图片
            current_group = {str(path1)}  # 使用集合存储相似图片
            
            for path2 in image_files:
                if str(path2) not in processed and str(path2) != str(path1):
                    similarity = self.compute_image_similarity(str(path1), str(path2))
                    if similarity > similarity_threshold:
                        current_group.add(str(path2))
                        print(f"\nFound similar: {path1} <-> {path2} (similarity: {similarity:.3f})")
            
            # 如果找到相似图片
            if len(current_group) > 1:
                group_images = []
                print(f"\nProcessing group with {len(current_group)} images:")
                for path in current_group:
                    score, metrics = self.get_image_quality_score(path)
                    group_images.append((path, score, metrics))
                    processed.add(path)
                    print(f"  - {path} (score: {score:.1f})")
                
                # 保存到 image_hashes 和 similar_groups
                self.image_hashes[str(path1)] = group_images
                similar_groups.append(group_images)
                
                # 添加到duplicates列表（除了最高质量的图片）
                sorted_images = sorted(group_images, key=lambda x: x[1], reverse=True)
                self.duplicates.extend([img[0] for img in sorted_images[1:]])
                
                print(f"Added group {len(similar_groups)} with {len(group_images)} images")
        
        print("\nImage comparison complete!")
        print(f"Total groups found: {len(similar_groups)}")
        for i, group in enumerate(similar_groups, 1):
            print(f"\nGroup {i} ({len(group)} images):")
            for path, score, _ in group:
                print(f"  - {path} (score: {score:.1f})")
        
        # 显示所有相似图片组
        if similar_groups:
            print(f"\nFound {len(similar_groups)} groups of similar images.")
            print("Use left/right arrow keys to navigate between groups.")
            self.visualizer.show_all_groups(similar_groups)
            
            # 保存duplicates文件
            output_file = self.save_duplicate_list()
            print(f"\nFound {len(self.duplicates)} duplicate images.")
            print(f"Duplicate list saved to: {output_file}")
        else:
            print("No duplicate images found.")

    def save_duplicate_list(self):
        """保存重复图片列表"""
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
        """根据图片内容动态调整权重"""
        weights = {
            'technical': 0.3,
            'face': 0.3,
            'aesthetic': 0.2,
            'scene': 0.2
        }
        
        # 根据场景类型调整
        if image_metrics['scene_type'] == 'portrait':
            weights['face'] = 0.4
            weights['technical'] = 0.2
        elif image_metrics['scene_type'] == 'landscape':
            weights['aesthetic'] = 0.3
            weights['face'] = 0.1
        
        return weights

    # ... (rest of the ImageDeduplicator class methods) 