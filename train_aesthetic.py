import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class AestheticDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        初始化数据集
        csv_file: 包含图片路径和评分的CSV文件
        img_dir: 图片目录
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        score = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(score, dtype=torch.float32)

def train_aesthetic_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.0001):
    """训练美学评分模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建 MobileNet 模型
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model = model.to(device)

    # 数据集和数据加载器
    train_dataset = AestheticDataset(
        csv_file=os.path.join(data_dir, 'train.csv'),
        img_dir=os.path.join(data_dir, 'images')
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 记录训练过程
    history = {
        'loss': [],
        'val_loss': []
    }

    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, scores in progress_bar:
            images = images.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.4f}')
        
        # 更新学习率
        scheduler.step(epoch_loss)

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/aesthetic_mobilenet.pth')
            print(f'Saved new best model with loss: {best_loss:.4f}')

    # 绘制训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

    return model, history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train aesthetic model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing images and train.csv')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    
    args = parser.parse_args()
    model, history = train_aesthetic_model(
        args.data_dir, 
        args.epochs, 
        args.batch_size,
        args.lr
    ) 