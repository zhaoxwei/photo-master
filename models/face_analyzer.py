import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import numpy as np
from PIL import Image
import time

class LightFaceAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            select_largest=False
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def analyze(self, image_path):
        try:
            start_time = time.time()
            
            # Load and detect faces
            img = Image.open(image_path)
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                return 0, [], time.time() - start_time
            
            face_score = 0
            face_analysis = []
            
            # Analyze each detected face
            for box, prob in zip(boxes, probs):
                if prob > 0.9:  # The confidence threshold is lowered from 0.98 to 0.9
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_img = img.crop((x1, y1, x2, y2))
                    
                    # Calculate face quality based on size and position
                    face_size = (x2 - x1) * (y2 - y1)
                    img_size = img.size[0] * img.size[1]
                    size_score = face_size / img_size * 100
                    
                    # Center position bonus
                    center_x = (x1 + x2) / 2 / img.size[0]
                    center_y = (y1 + y2) / 2 / img.size[1]
                    position_score = (1 - abs(0.5 - center_x) - abs(0.5 - center_y)) * 50
                    
                    # Combined quality score
                    quality = size_score + position_score
                    face_score += quality
                    
                    face_analysis.append({
                        'confidence': float(prob),
                        'box': box.tolist(),
                        'quality': float(quality),
                        'size_score': float(size_score),
                        'position_score': float(position_score)
                    })
            
            # Bonus for group photos (but not too many faces)
            if len(face_analysis) > 1:
                face_score *= min(len(face_analysis), 5)
            
            process_time = time.time() - start_time
            return face_score, face_analysis, process_time
            
        except Exception as e:
            print(f"Face analysis error for {image_path}: {str(e)}")
            return 0, [], 0 

class EnhancedFaceAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # Add a facial expression recognition model
        self.emotion_model = EmotionNet().to(self.device)
        # Add a facial pose estimation model
        self.pose_estimator = PoseEstimator().to(self.device)
        
    def analyze(self, image_path):
        # Existing facial detection
        boxes, probs = self.mtcnn.detect(img)
        
        face_score = 0
        for box, prob in zip(boxes, probs):
            # Base score (current)
            size_score = self.calculate_size_score(box, img.size)
            position_score = self.calculate_position_score(box, img.size)
            
            # Add new scoring items
            emotion_score = self.analyze_emotion(face_img)  # Facial expression score
            pose_score = self.analyze_pose(face_img)       # Pose score
            blur_score = self.detect_blur(face_img)        # Blur detection
            eyes_score = self.check_eyes_open(face_img)    # Eye openness detection
            
            # Overall score
            face_quality = ( 
                0.3 * size_score +
                0.2 * position_score +
                0.2 * emotion_score +
                0.15 * pose_score +
                0.1 * blur_score +
                0.05 * eyes_score
            ) 