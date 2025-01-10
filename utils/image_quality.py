import cv2
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import numpy as np
from models.face_analyzer import LightFaceAnalyzer
from models.aesthetic_analyzer import AestheticAnalyzer

class ImageQualityAnalyzer:
    def __init__(self):
        """Initialize image quality analyzer"""
        self.aesthetic_analyzer = AestheticAnalyzer()
        self.face_analyzer = LightFaceAnalyzer()
        
        # Weight configuration
        self.weights = {
            'resolution': 0.25,
            'clarity': 0.25,
            'face_score': 0.25,
            'aesthetic': 0.25
        }

    def analyze_image(self, image_path):
        """Analyze image quality"""
        try:
            # 1. Technical score
            resolution = self.get_resolution_score(image_path)
            
            # 2. Clarity score
            clarity = self.get_clarity_score(image_path)
            
            # 3. Face score
            face_score, face_analysis, face_time = self.face_analyzer.analyze(image_path)
            
            # 4. Aesthetic score
            aesthetic_result = self.aesthetic_analyzer.analyze(image_path)
            aesthetic_score = aesthetic_result.get('aesthetic_score', 0)
            
            # Normalize scores
            normalized_scores = {
                'resolution': resolution,
                'clarity': clarity,
                'face_score': face_score,
                'aesthetic': aesthetic_score
            }
            
            # Calculate total score
            total_score = sum(score * self.weights[metric] 
                            for metric, score in normalized_scores.items())
            
            # Return detailed results
            return total_score, {
                'resolution': resolution,
                'clarity': clarity,
                'face_score': face_score,
                'face_analysis': face_analysis,
                'aesthetic_score': aesthetic_score,
                'normalized_scores': normalized_scores,
                'timing': {
                    'face_analysis': face_time,
                    'aesthetic_analysis': aesthetic_result.get('timing', {})
                }
            }
            
        except Exception as e:
            print(f"Error analyzing image quality: {str(e)}")
            return 0, {}

    def get_resolution_score(self, image_path):
        """Calculate resolution score"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0
            
            height, width = img.shape[:2]
            mp = (width * height) / 1_000_000  # Convert to million pixels
            
            # Scoring rules:
            # 12MP = 80 points
            # 24MP = 90 points
            # 48MP = 95 points
            if mp <= 12:
                score = 80 * (mp / 12)
            elif mp <= 24:
                score = 80 + 10 * ((mp - 12) / 12)
            elif mp <= 48:
                score = 90 + 5 * ((mp - 24) / 24)
            else:
                score = 95 + 5 * min(1, (mp - 48) / 48)
            
            return min(100, score)
            
        except Exception as e:
            print(f"Error calculating resolution score: {str(e)}")
            return 0

    def get_clarity_score(self, image_path):
        """Calculate clarity score"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate Sobel gradient
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_mean = np.mean(sobel_mag)
            
            # Overall score
            clarity_score = min(100, (laplacian_var / 500) * 50 + (sobel_mean / 50) * 50)
            
            return clarity_score
            
        except Exception as e:
            print(f"Error calculating clarity score: {str(e)}")
            return 0 

    def compute_hash(self, image_path):
        """Compute image hash"""
        try:
            return str(imagehash.average_hash(Image.open(image_path)))
        except Exception as e:
            print(f"Error computing hash for {image_path}: {str(e)}")
            return None 