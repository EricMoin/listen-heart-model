import os
import json
from main import VoiceMoodTreeHole

def test_system():
    """
    测试情绪树洞语音助手系统
    """
    # 检查模型文件是否存在
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/emotion_model.pth"))
    print(f"使用模型文件路径: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在，请先运行 train_model.py 训练模型")
        return
    
    # 创建情绪树洞语音助手实例
    print("初始化情绪树洞语音助手...")
    assistant = VoiceMoodTreeHole()
    
    # 获取测试目录中的所有音频文件
    test_dir = "../test"
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
    
    if not test_files:
        print(f"错误：测试目录 {test_dir} 中没有找到 .wav 音频文件")
        return
    
    # 处理每个测试音频文件
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        print(f"\n处理测试音频: {test_file}")
        
        # 处理音频并获取结果
        result = assistant.process_audio(test_path)
        
        # 打印结果
        print("\n处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n" + "-"*50)

if __name__ == "__main__":
    test_system()