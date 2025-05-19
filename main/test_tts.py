import os
import sys
from text_to_speech import TextToSpeech
from language_model import LanguageModel

def main():
    """
    测试大模型生成文本并转换为语音的流程
    """
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在初始化大语言模型...")
    language_model = LanguageModel(use_deepseek=False)  # 使用Qwen_1.8B模型
    
    print("正在初始化TTS模型...")
    tts = TextToSpeech()
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("请输入问题: ")
    
    print(f"\n用户问题: {query}")
    
    # 生成回复
    print("正在生成回复...")
    response = language_model.generate_response(query, "中性", "正常")
    print(f"\n大模型回复: {response}")
    
    # 将回复转换为语音
    print("\n正在将回复转换为语音...")
    output_path = tts.synthesize_speech(response, output_filename=f"response_{int(hash(query) % 10000):04d}")
    
    if output_path:
        print(f"\n成功生成语音! 文件保存在: {output_path}")
    else:
        print("\n语音生成失败")

if __name__ == "__main__":
    main() 