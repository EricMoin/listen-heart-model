import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechToText:
    """
    语音转文本类，使用Whisper模型
    """
    def __init__(self, model_name="openai/whisper-small"):
        """
        初始化Whisper模型
        """
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None
        
        # 如果有GPU则使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"语音转文本模型已加载到设备: {self.device}")
    
    def transcribe(self, audio_path, language="chinese"):
        """
        将音频文件转换为文本
        """
        try:
            # 加载音频
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            
            # 处理音频
            input_features = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            
            # 生成文本
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
            # 解码文本
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription
        
        except Exception as e:
            print(f"转录音频时出错: {e}")
            return ""

if __name__ == "__main__":
    # 示例用法
    stt = SpeechToText()
    audio_path = "../test/sad.wav"
    text = stt.transcribe(audio_path)
    print(f"转录结果: {text}")