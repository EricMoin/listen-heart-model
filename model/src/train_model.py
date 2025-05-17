import os
import numpy as np
import pandas as pd
import torch
from audio_preprocessing import process_ravdess_dataset, save_features
from emotion_recognition_model import prepare_data, EmotionRecognitionModel, train_model, save_model

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "ravdess")
    features_path = os.path.join(base_dir, "output", "ravdess_features.pkl")
    model_path = os.path.join(base_dir, "models", "emotion_model.pth")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 检查特征文件是否存在
    if not os.path.exists(features_path):
        print("正在处理RAVDESS数据集并提取特征...")
        features_df = process_ravdess_dataset(data_path)
        save_features(features_df, features_path)
        print("特征提取完成并保存。")
    else:
        print(f"加载已有特征文件: {features_path}")
        features_df = pd.read_pickle(features_path)
        
        # 检查特征格式
        print("\n检查特征格式...")
        sample_feature = features_df['features'].iloc[0]
        
        # 如果特征不是字典格式，需要重新处理数据
        if not isinstance(sample_feature, dict):
            print("\n特征格式不正确，需要重新处理数据...")
            features_df = process_ravdess_dataset(data_path)
            save_features(features_df, features_path)
            print("数据已重新处理并保存")
    
    print(f"\n数据集大小: {len(features_df)} 个样本")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备训练和验证数据...")
    train_loader, val_loader, emotion_encoder, intensity_encoder, input_dim = prepare_data(features_df)
    
    # 创建模型
    print("创建情绪识别模型...")
    hidden_dim = 128
    emotion_classes = len(emotion_encoder.classes_)
    intensity_classes = len(intensity_encoder.classes_)
    
    print(f"输入维度 (MEL特征): {input_dim}")
    print(f"情绪类别数: {emotion_classes}")
    print(f"强度类别数: {intensity_classes}")
    
    model = EmotionRecognitionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        emotion_classes=emotion_classes,
        intensity_classes=intensity_classes
    ).to(device)
    
    # 训练模型
    print("开始训练模型...")
    trained_model = train_model(model, train_loader, val_loader, device, epochs=200)
    
    # 保存模型
    save_model(trained_model, emotion_encoder, intensity_encoder, model_path)
    print(f"模型训练完成，已保存到: {model_path}")

if __name__ == "__main__":
    main()