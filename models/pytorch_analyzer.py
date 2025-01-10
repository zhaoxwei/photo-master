import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

class PytorchAnalyzer:
    def __init__(self, model_type='mobilenet'):
        """
        Initialize PyTorch analyzer
        model_type: 'mobilenet'
        """
        if model_type != 'mobilenet':
            print("Warning: Only MobileNet model is supported, using MobileNet.")
            model_type = 'mobilenet'
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = self._load_model()
        self.transform = self._get_transforms()
        print(f"PyTorch analyzer initialized with MobileNet on {self.device}")

    def _load_model(self):
        print("\nLoading MobileNet model...")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 1)
        print("MobileNet V2 architecture loaded")
        
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """Obtain image preprocessing transformations"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def analyze(self, image_path):
        """Analyze image"""
        try:
            # Start timing
            start_time = time.time()
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Model inference
            output = self.model(input_tensor)
            score = float(torch.sigmoid(output).item() * 100)  # Convert to 0-100
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            return {
                'score': score,
                'process_time': process_time,
                'model_type': self.model_type
            }
            
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return {
                'score': 0,
                'error': str(e),
                'model_type': self.model_type
            }

    def batch_analyze(self, image_paths, batch_size=4):
        """Batch analyze images"""
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    continue
            
            if not batch_tensors:
                continue
                
            # Process batch
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                scores = torch.sigmoid(outputs).cpu().numpy() * 100
            
            # Save results
            for path, score in zip(batch_paths, scores):
                results.append({
                    'path': path,
                    'score': float(score),
                    'model_type': self.model_type
                })
        
        return results 