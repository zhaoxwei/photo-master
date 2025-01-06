from models.aesthetic_analyzer import AestheticAnalyzer
import time
import os

def test_model(model_type, image_path):
    print(f"\nTesting {model_type} model...")
    analyzer = AestheticAnalyzer(model_type=model_type)
    
    start_time = time.time()
    result = analyzer.analyze(image_path)
    process_time = time.time() - start_time
    
    print(f"Results for {os.path.basename(image_path)}:")
    print(f"Score: {result['aesthetic_score']:.1f}")
    print(f"Processing time: {process_time:.3f}s")
    print(f"Model type: {result['model_type']}")
    print("-" * 50)
    return result

def compare_models(image_path):
    """比较两个模型的结果"""
    print(f"\nComparing models on: {os.path.basename(image_path)}")
    
    mobilenet_result = test_model('mobilenet', image_path)
    resnet_result = test_model('resnet50', image_path)
    
    print("\nComparison Summary:")
    print(f"MobileNet score: {mobilenet_result['aesthetic_score']:.1f}")
    print(f"ResNet50 score: {resnet_result['aesthetic_score']:.1f}")
    print(f"Score difference: {abs(mobilenet_result['aesthetic_score'] - resnet_result['aesthetic_score']):.1f}")

def main():
    # 使用示例图片进行测试
    test_images = [
        "test_images/landscape.jpg",
        "test_images/portrait.jpg",
        # 添加更多测试图片...
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            compare_models(image_path)
        else:
            print(f"Warning: Test image not found: {image_path}")

if __name__ == '__main__':
    main() 