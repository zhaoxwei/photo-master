import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

class PytorchAnalyzer:
    def __init__(self, model_type='mobilenet'):
        """
        初始化 PyTorch 分析器
        model_type: 只支持 'mobilenet'
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
        """加载并配置模型"""
        print("\nLoading MobileNet model...")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 1)
        print("MobileNet V2 architecture loaded")
        
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """获取图像预处理转换"""
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
        """分析图片"""
        try:
            # 计时开始
            start_time = time.time()
            
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 模型推理
            output = self.model(input_tensor)
            score = float(torch.sigmoid(output).item() * 100)  # 转换为0-100分
            
            # 计算处理时间
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
        """批量分析图片"""
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
                
            # 处理批次
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                scores = torch.sigmoid(outputs).cpu().numpy() * 100
            
            # 保存结果
            for path, score in zip(batch_paths, scores):
                results.append({
                    'path': path,
                    'score': float(score),
                    'model_type': self.model_type
                })
        
        return results 