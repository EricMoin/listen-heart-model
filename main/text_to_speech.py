import os
import sys
import torch
import numpy as np

# 将fastspeech路径添加到系统路径
fastspeech_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastspeech")
sys.path.append(fastspeech_dir)

# 导入fastspeech模块
from synthesize_all import SpeechSynthesis

class TextToSpeech:
    """
    文本转语音类，使用Chinese-FastSpeech2模型
    """
    _model = None
    _model_loaded = False
    _original_cwd = os.getcwd()  # 保存原始工作目录
    
    @classmethod
    def load_model(cls, force_reload=False):
        """
        加载TTS模型
        """
        if cls._model_loaded and not force_reload:
            print("TTS模型已加载，无需重新加载")
            return
            
        print("正在加载TTS模型...")
        
        try:
            # 保存原始工作目录
            cls._original_cwd = os.getcwd()
            
            # 切换到fastspeech目录，确保相对路径正确解析
            fastspeech_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastspeech")
            print(f"切换工作目录到: {fastspeech_dir}")
            os.chdir(fastspeech_dir)
            
            # 加载Chinese-FastSpeech2模型
            config_path = os.path.join(fastspeech_dir, "config", "AISHELL3")
            print(f"使用配置路径: {config_path}")
            cls._model = SpeechSynthesis(config_path)
            
            # 加载完成后恢复原始工作目录
            os.chdir(cls._original_cwd)
            print(f"恢复工作目录到: {cls._original_cwd}")
            
            cls._model_loaded = True
            print("TTS模型加载完成")
            
        except Exception as e:
            # 发生异常时也要恢复工作目录
            os.chdir(cls._original_cwd)
            print(f"恢复工作目录到: {cls._original_cwd}")
            print(f"加载TTS模型时出错: {str(e)}")
            raise
    
    @classmethod
    def unload_model(cls):
        """
        卸载TTS模型
        """
        if not cls._model_loaded:
            print("TTS模型未加载，无需卸载")
            return
            
        print("正在卸载TTS模型...")
        cls._model = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        cls._model_loaded = False
        print("TTS模型卸载完成")
    
    def __init__(self):
        # 确保模型已加载
        if not TextToSpeech._model_loaded:
            TextToSpeech.load_model()
        
        # 初始化输出目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def synthesize_speech(self, text, output_filename=None):
        """
        将文本转换为语音
        
        Args:
            text (str): 要转换的文本
            output_filename (str, optional): 输出文件名，如果不指定则自动生成
            
        Returns:
            str: 生成的音频文件路径
        """
        if not TextToSpeech._model_loaded:
            TextToSpeech.load_model()
            
        try:
            if output_filename is None:
                # 生成默认输出文件名
                output_filename = f"tts_output_{int(np.random.random() * 1000000)}"
            
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 保存当前工作目录
            current_cwd = os.getcwd()
            
            # 切换到fastspeech目录，确保相对路径正确解析
            fastspeech_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastspeech")
            os.chdir(fastspeech_dir)
            
            # 生成语音
            print(f"将文本转换为语音: {text}")
            save_path = self.output_dir
            output_path = TextToSpeech._model.text2speech(text, save_path)
            
            # 恢复工作目录
            os.chdir(current_cwd)
            
            # 如果输出路径是带有tmp名称的临时文件，重命名为指定的名称
            if os.path.basename(output_path) == "tmp.wav":
                new_path = os.path.join(self.output_dir, f"{output_filename}.wav")
                
                # 检查目标文件是否存在，如果存在则先删除
                if os.path.exists(new_path):
                    print(f"目标文件已存在，正在删除: {new_path}")
                    os.remove(new_path)
                
                print(f"重命名文件: {output_path} -> {new_path}")
                os.rename(output_path, new_path)
                output_path = new_path
            
            print(f"语音生成完成: {output_path}")
            return output_path
            
        except Exception as e:
            # 确保恢复工作目录
            if 'current_cwd' in locals():
                os.chdir(current_cwd)
            print(f"生成语音时出错: {str(e)}")
            return None


def main():
    """
    测试函数
    """
    import sys
    
    # 创建TTS实例
    tts = TextToSpeech()
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = input("请输入要转换的文本: ")
    
    # 生成语音
    output_path = tts.synthesize_speech(text)
    
    if output_path:
        print(f"语音已保存到: {output_path}")
    else:
        print("语音生成失败")


if __name__ == "__main__":
    main() 