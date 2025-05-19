import os
import json
import numpy as np
import torch
from audio_preprocessing import load_audio, extract_features, load_wav2vec2
from emotion_recognition_model import load_model
from speech_to_text import SpeechToText
from language_model import LanguageModel
from text_to_speech import TextToSpeech

class VoiceMoodTreeHole:
    """
    情绪树洞语音助手主类
    """
    # 类变量，用于存储已加载的模型
    _emotion_model = None
    _emotion_encoder = None
    _intensity_encoder = None
    _stt_model = None
    _language_model = None
    _tts_model = None
    _device = None
    _models_loaded = False
    
    @classmethod
    def load_models(cls, force_reload=False):
        """
        加载所有模型到内存中
        """
        if cls._models_loaded and not force_reload:
            print("模型已加载，无需重新加载")
            return
            
        print("正在加载模型到内存...")
        # 设置设备
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置模型路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "emotion_model.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 加载情绪识别模型
            input_dim = 128  # 根据特征提取函数的输出维度设置
            hidden_dim = 128
            cls._emotion_model, cls._emotion_encoder, cls._intensity_encoder = load_model(
                model_path, input_dim, hidden_dim, force_input_dim=True
            )
            
            # 将模型移动到设备并设置为全精度
            cls._emotion_model = cls._emotion_model.to(cls._device)
            # 不使用半精度，保持全精度以获得更好的性能
            cls._emotion_model.eval()
            
            # 初始化语音转文本模型
            print("正在加载语音转文本模型...")
            cls._stt_model = SpeechToText()
            
            # 初始化大语言模型
            print("正在加载大语言模型...")
            cls._language_model = LanguageModel(use_deepseek=False)  # 使用Qwen_1.8B模型
            
            # 初始化TTS模型
            print("正在加载TTS模型...")
            cls._tts_model = TextToSpeech()
            
            cls._models_loaded = True
            print("所有模型已加载完成")
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
    
    @classmethod
    def unload_models(cls):
        """
        从内存中卸载模型
        """
        if not cls._models_loaded:
            print("模型未加载，无需卸载")
            return
            
        print("正在卸载模型...")
        cls._emotion_model = None
        cls._emotion_encoder = None
        cls._intensity_encoder = None
        cls._stt_model = None
        cls._language_model = None
        cls._tts_model = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        cls._models_loaded = False
        print("所有模型已卸载完成")
    
    def __init__(self):
        # 初始化路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "emotion_model.pth")
        self.output_dir = os.path.join(base_dir, "output")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 确保模型已加载
        if not VoiceMoodTreeHole._models_loaded:
            VoiceMoodTreeHole.load_models()
    
    def predict_emotion(self, audio_path):
        """
        预测音频的情绪和强度
        """
        # 确保模型已加载
        if not VoiceMoodTreeHole._models_loaded:
            VoiceMoodTreeHole.load_models()
            
        try:
            # 加载音频 - 使用16kHz采样率（wav2vec2的要求）
            audio_data, sample_rate = load_audio(audio_path, sr=16000)
            if audio_data is None:
                return None, None
            
            # 提取MEL特征 - 使用原始采样率
            mel_features = extract_features(audio_data, sample_rate=16000)
            
            # 提取wav2vec特征 - 在GPU上运行（如果可用）
            processor, wav2vec_model = load_wav2vec2()
            wav2vec_model.eval()  # 模型已经在load_wav2vec2函数中移至GPU（如果可用）
            
            with torch.no_grad():
                # 处理音频数据 - 在GPU上运行（如果可用）
                inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                # 如果有GPU，将输入移至GPU
                if torch.cuda.is_available():
                    inputs = {k: v.to(VoiceMoodTreeHole._device) for k, v in inputs.items()}
                outputs = wav2vec_model(**inputs)
                # 保持张量格式，不转换为numpy
                wav2vec_features = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # 构造特征字典
            features = {
                'mel': torch.FloatTensor(mel_features).unsqueeze(0).to(VoiceMoodTreeHole._device),
                'wav2vec': wav2vec_features.unsqueeze(0).to(VoiceMoodTreeHole._device) if torch.is_tensor(wav2vec_features) else torch.FloatTensor(wav2vec_features).unsqueeze(0).to(VoiceMoodTreeHole._device)
            }
            
            # 预测 - 在GPU上运行
            with torch.no_grad():
                emotion_output, intensity_output = VoiceMoodTreeHole._emotion_model(features)
                
                # 获取预测类别
                _, emotion_pred = torch.max(emotion_output, 1)
                _, intensity_pred = torch.max(intensity_output, 1)
                
                # 解码预测结果
                emotion_code = VoiceMoodTreeHole._emotion_encoder.inverse_transform(emotion_pred.cpu().numpy())[0]
                intensity_code = VoiceMoodTreeHole._intensity_encoder.inverse_transform(intensity_pred.cpu().numpy())[0]
                
                # 转换为文字标签
                emotion_map = {
                    "01": "中性",
                    "02": "平静",
                    "03": "开心",
                    "04": "悲伤",
                    "05": "愤怒",
                    "06": "恐惧",
                    "07": "厌恶",
                    "08": "惊讶"
                }
                
                intensity_map = {
                    "01": "正常",
                    "02": "强烈"
                }
                
                emotion_text = emotion_map.get(emotion_code, "未知")
                intensity_text = intensity_map.get(intensity_code, "未知")
            
            return emotion_text, intensity_text
            
        except Exception as e:
            print(f"预测情绪时出错: {str(e)}")
            return None, None
    
    def transcribe_audio(self, audio_path):
        """
        将音频转换为文本
        """
        # 确保模型已加载
        if not VoiceMoodTreeHole._models_loaded:
            VoiceMoodTreeHole.load_models()
            
        return VoiceMoodTreeHole._stt_model.transcribe(audio_path)
    
    def generate_response(self, text, emotion, intensity):
        """
        生成回应
        """
        # 确保模型已加载
        if not VoiceMoodTreeHole._models_loaded:
            VoiceMoodTreeHole.load_models()
            
        return VoiceMoodTreeHole._language_model.generate_response(text, emotion, intensity)
    
    def process_audio(self, audio_path):
        """
        处理音频文件并生成回应
        """
        print(f"正在处理音频: {audio_path}")
        
        # 预测情绪和强度
        print("正在分析情绪...")
        emotion, intensity = self.predict_emotion(audio_path)
        if emotion is None:
            return {"error": "无法处理音频文件"}
        
        # 转录音频
        print("正在转录语音...")
        text = self.transcribe_audio(audio_path)
        if not text:
            return {"error": "无法转录音频文件"}
        
        # 生成回应
        print("正在生成回应...")
        response = self.generate_response(text, emotion, intensity)
        
        # 生成语音回应
        print("正在生成语音回应...")
        response_audio_path = None
        try:
            response_audio_path = VoiceMoodTreeHole._tts_model.synthesize_speech(
                response,
                output_filename=f"{os.path.basename(audio_path).split('.')[0]}_response"
            )
        except Exception as e:
            print(f"生成语音回应时出错: {str(e)}")
            print("继续处理，但不生成语音")
        
        # 构造结果
        result = {
            "text": text,
            "emotion": emotion,
            "intensity": intensity,
            "response": response
        }
        
        # 如果成功生成语音，添加到结果中
        if response_audio_path:
            result["response_audio"] = response_audio_path
        
        # 保存结果
        output_file = os.path.join(self.output_dir, f"{os.path.basename(audio_path).split('.')[0]}_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存到: {output_file}")
        
        return result

def main():
    print("情绪树洞语音助手 - 模型常驻内存版")
    print("==================================")
    print("1. 预加载模型")
    print("2. 处理音频")
    print("3. 卸载模型")
    print("4. 退出")
    print("==================================")
    
    # 设置基础路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_audio = os.path.join(base_dir, "test", "sad.wav")
    
    while True:
        choice = input("\n请选择操作 (1-4): ")
        
        if choice == "1":
            try:
                # 预加载模型
                VoiceMoodTreeHole.load_models(force_reload=True)
            except FileNotFoundError as e:
                print(f"\n错误: {e}")
                print("请确保已经训练并保存了模型。")
                print("可以运行 'python src/train_model.py' 来训练模型。")
            except Exception as e:
                print(f"\n加载模型时出错: {e}")
            
        elif choice == "2":
            # 处理音频
            custom_path = input(f"请输入音频文件路径 (默认: {test_audio}): ")
            if custom_path.strip():
                test_audio = custom_path
                
            if not os.path.exists(test_audio):
                print(f"\n错误: 音频文件不存在: {test_audio}")
                continue
                
            try:
                # 创建情绪树洞语音助手实例
                assistant = VoiceMoodTreeHole()
                
                # 处理音频
                result = assistant.process_audio(test_audio)
                
                # 打印结果
                print("\n处理结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"\n处理音频时出错: {e}")
            
        elif choice == "3":
            # 卸载模型
            VoiceMoodTreeHole.unload_models()
            
        elif choice == "4":
            # 退出
            print("正在退出...")
            break
            
        else:
            print("无效的选择，请重新输入")

if __name__ == "__main__":
    main()