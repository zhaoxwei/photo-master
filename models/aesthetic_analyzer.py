from models.pytorch_analyzer import PytorchAnalyzer
import time

class AestheticAnalyzer:
    def __init__(self, model_type='mobilenet'):
        """
        Initialize aesthetic analyzer
        model_type: only supports 'mobilenet'
        """
        if model_type != 'mobilenet':
            print("Warning: Only MobileNet model is supported, using MobileNet.")
            model_type = 'mobilenet'
        
        self.model_type = model_type
        self.analyzer = PytorchAnalyzer(model_type=model_type)

    def analyze(self, image_path):
        """Analyze image aesthetic quality"""
        start_time = time.time()
        
        # Get aesthetic score
        result = self.analyzer.analyze(image_path)
        aesthetic_score = result['score']
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        return {
            'aesthetic_score': aesthetic_score,
            'model_type': self.model_type,
            'timing': {
                'model_inference': result['process_time'],
                'total_time': total_time
            }
        }

    def batch_analyze(self, image_paths, batch_size=4):
        """Batch analyze images"""
        return self.analyzer.batch_analyze(image_paths, batch_size) 