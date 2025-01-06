# PhotoMaster

A deep learning based smart photo deduplication and curation tool using MobileNet for aesthetic scoring.

## Technology Stack

### Deep Learning Framework
- PyTorch 1.8+
- torchvision
- CUDA support (optional)

### Computer Vision
- OpenCV (cv2)
- scikit-image
- Pillow (PIL)

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib
- tqdm (progress display)

## Core Models

### 1. MobileNet V2
- Purpose: Aesthetic quality scoring
- Architecture: Depth-wise separable convolution
- Pre-trained: ImageNet
- Parameters: ~3.5M
- Input: 224x224 RGB image
- Output: Aesthetic score (0-100)

### 2. MTCNN (Multi-task Cascaded CNN)
- Purpose: Face detection and analysis
- Architecture: Three-stage cascade CNN
  - P-Net: Initial detection
  - R-Net: Bounding box regression
  - O-Net: Landmark localization
- Output:
  - Face bounding boxes
  - Facial landmarks
  - Confidence scores

### 3. Image Similarity Algorithms
- pHash (Perceptual Hash)
- SSIM (Structural Similarity)
- Color Histogram Comparison
- Local Feature Matching

## Features

- Automatic similar photo detection
- Multi-dimensional quality scoring
  - Technical metrics (resolution)
  - Clarity assessment
  - Face detection and analysis
  - Aesthetic scoring
- Multiple image format support (JPG, JPEG, PNG, BMP, TIFF)
- Interactive visualization interface
- Batch processing support

## Project Structure

```text
project/
├── main.py                    # Main program entry
├── train_aesthetic.py         # Aesthetic scoring model training
├── models/
│   ├── aesthetic_analyzer.py  # Aesthetic analyzer
│   ├── pytorch_analyzer.py    # PyTorch model wrapper
│   └── face_analyzer.py       # Face analyzer
└── utils/
    ├── image_quality.py       # Image quality analysis
    └── visualization.py       # Visualization tools
```

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/photo-master.git
cd photo-master
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py
```

### Advanced Options
```bash
# Adjust similarity threshold (default: 0.65)
python main.py --threshold 0.7
```

## How It Works

### Similarity Analysis
- Scene hash comparison (35%): Overall scene similarity
- Color distribution (25%): Color histogram analysis
- Structural similarity (30%): SSIM algorithm
- Local feature comparison (10%): Local region analysis

### Quality Scoring System
1. Technical Score
   - Resolution based (MP value)
   - 12MP = 80 points
   - 24MP = 90 points
   - 48MP = 95 points

2. Clarity Score
   - Laplacian edge detection
   - Sobel gradient analysis
   - Local contrast evaluation

3. Face Analysis
   - Face detection and localization
   - Face size evaluation
   - Multi-face scene bonus

4. Aesthetic Score
   - Using MobileNet model
   - Composition and color evaluation
   - Based on ImageNet pre-training

## Output

1. Visual Interface
   - Similar photo group display
   - Detailed scoring for each photo
   - Keyboard navigation (arrow keys)
   - Toolbar navigation buttons

2. JSON Report
   - Filename: duplicates_timestamp.json
   - Similar image group information
   - Detailed quality scoring data

## Performance Optimization

- Image preprocessing and caching
- Multi-process parallel processing
- GPU acceleration support (if available)
- Batch processing optimization
- Memory usage optimization

## Notes

1. Runtime Environment
   - Python 3.7+
   - CUDA support (optional, for GPU acceleration)
   - Sufficient system memory (8GB+ recommended)

2. Performance Considerations
   - Recommended photo resolution: up to 4K
   - Large batch processing may take time
   - GPU significantly improves speed

3. Known Issues
   - OpenMP runtime conflict handled
   - Some RAW formats not supported
   - Large images may require more memory

## FAQ

Q: Why is the program running slowly?
A: Try reducing the number of photos or use GPU acceleration.

Q: What similarity threshold should I use?
A: Default is 0.65, recommended range 0.55-0.75, higher is stricter.

Q: Why are some photos not detected?
A: They might be below the similarity threshold, try lowering it.

## License

MIT License

## Changelog

### v1.0.0
- Initial release
- Basic functionality
- MobileNet model support

---

[Chinese version follows]

# 智能照片去重工具

一个基于深度学习的智能照片去重和筛选工具，使用 MobileNet 模型进行美学评分。

## 技术框架

### 深度学习框架
- PyTorch 1.8+
- torchvision
- CUDA支持（可选）

### 计算机视觉
- OpenCV (cv2)
- scikit-image
- Pillow (PIL)

### 数据处理
- NumPy
- Pandas

### 可视化
- Matplotlib
- tqdm（进度显示）

## 核心模型

### 1. MobileNet V2
- 用途：美学质量评分
- 架构：深度可分离卷积
- 预训练：ImageNet
- 参数量：约3.5M
- 输入：224x224 RGB图像
- 输出：美学评分（0-100）

### 2. MTCNN (多任务级联CNN)
- 用途：人脸检测和分析
- 架构：三阶段级联CNN
  - P-Net：初步检测
  - R-Net：边界框回归
  - O-Net：关键点定位
- 输出：
  - 人脸边界框
  - 面部关键点
  - 置信度分数

### 3. 图像相似度算法
- pHash（感知哈希）
- SSIM（结构相似性）
- 颜色直方图比较
- 局部特征匹配

## 功能特点

- 自动检测相似照片
- 多维度质量评分
  - 技术指标评分（分辨率）
  - 清晰度评分
  - 人脸检测和分析
  - 美学评分
- 支持多种图片格式（JPG、JPEG、PNG、BMP、TIFF）
- 交互式可视化界面
- 批量处理支持

## 项目结构

```text
project/
├── main.py                    # 主程序入口
├── train_aesthetic.py         # 美学评分模型训练脚本
├── models/
│   ├── aesthetic_analyzer.py  # 美学分析器
│   ├── pytorch_analyzer.py    # PyTorch 模型封装
│   └── face_analyzer.py       # 人脸分析器
└── utils/
    ├── image_quality.py       # 图像质量分析
    └── visualization.py       # 可视化工具
```

## 核心文件说明

### 主程序
- `main.py`
  - 程序入口
  - 命令行参数处理
  - OpenMP 运行时配置

### 模型组件
- `models/pytorch_analyzer.py`
  - MobileNet V2 模型加载和推理
  - 图像预处理和后处理
  - GPU/CPU 设备管理

- `models/aesthetic_analyzer.py`
  - 美学评分分析器
  - 多维度质量评估
  - 性能计时和监控

- `models/face_analyzer.py`
  - MTCNN 人脸检测
  - 人脸特征分析
  - 多人脸场景处理

### 工具组件
- `utils/visualization.py`
  - 交互式可视化界面
  - 图片组导航功能
  - 评分信息显示
  - OpenMP 优化

- `utils/image_quality.py`
  - 技术指标分析
  - 清晰度评估
  - 图像相似度计算

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/photo-master.git
cd photo-master
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法
```bash
python main.py /path/to/photos
```

### 高级选项
```bash
# 调整相似度阈值（默认0.65）
python main.py /path/to/photos --threshold 0.7
```

## 工作原理

### 相似度分析
- 场景哈希比较 (35%)：检测整体场景相似性
- 颜色分布比较 (25%)：分析颜色直方图
- 结构相似度分析 (30%)：使用SSIM算法
- 局部特征比较 (10%)：分析图像局部区域

### 质量评分系统
1. 技术指标评分
   - 分辨率评分（基于MP值）
   - 12MP = 80分
   - 24MP = 90分
   - 48MP = 95分

2. 清晰度评分
   - Laplacian边缘检测
   - Sobel梯度分析
   - 局部对比度评估

3. 人脸分析
   - 人脸检测和定位
   - 人脸大小评估
   - 多人脸场景加分

4. 美学评分
   - 使用MobileNet模型
   - 评估构图和色彩
   - 基于ImageNet预训练

## 输出结果

1. 可视化界面
   - 相似照片分组展示
   - 每张照片的详细评分
   - 支持键盘导航（左右方向键）
   - 工具栏导航按钮

2. JSON报告
   - 文件名：duplicates_时间戳.json
   - 包含相似图片组信息
   - 详细的质量评分数据

## 性能优化

- 图像预处理和缓存机制
- 多进程并行处理
- GPU加速支持（如果可用）
- 批量处理优化
- 内存使用优化

## 注意事项

1. 运行环境
   - Python 3.7+
   - CUDA支持（可选，用于GPU加速）
   - 足够的系统内存（建议8GB以上）

2. 性能考虑
   - 建议处理照片分辨率不超过4K
   - 大量图片处理可能需要较长时间
   - 可用GPU会显著提升处理速度

3. 已知问题
   - OpenMP运行时冲突已自动处理
   - 某些RAW格式可能不支持
   - 超大图片可能需要更多内存

## 常见问题解答

Q: 程序运行很慢怎么办？
A: 可以尝试减少待处理图片数量，或使用GPU加速。

Q: 相似度阈值应该设置多少？
A: 默认0.65，建议范围0.55-0.75，值越大要求越严格。

Q: 为什么有些照片没有被检测出来？
A: 可能是相似度低于阈值，可以适当调低阈值重试。

## 许可证

MIT License

## 更新日志

### v1.0.0
- 初始版本发布
- 基本功能实现
- MobileNet模型支持