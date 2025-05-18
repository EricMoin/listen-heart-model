import os
import sys
from language_model import LanguageModel

def main():
    """测试微调后的Qwen模型集成"""
    print("开始测试微调后的Qwen模型集成...")
    
    # 初始化模型
    model = LanguageModel()
    
    # 测试不同情绪下的回应，使用更详细的输入
    test_inputs = [
        ("我今天感到很开心，因为我收到了期待已久的礼物", "happy"),
        ("我有点担心未来会怎样，感觉很迷茫", "fearful"),
        ("我对这个人真的很生气，他总是说话不算话", "angry"),
        ("我感觉很悲伤，好像什么都提不起兴趣", "sad"),
        ("这个事情让我很惊讶，我完全没有想到会这样", "surprised")
    ]
    
    print("\n" + "="*50)
    print("测试微调模型的回应质量")
    print("="*50 + "\n")
    
    for i, (text, emotion) in enumerate(test_inputs):
        print(f"测试 {i+1}:")
        print(f"用户输入 ({emotion}): {text}")
        response = model.generate_response(text, emotion, "medium")
        print(f"模型回应: {response}")
        print("-" * 50)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 