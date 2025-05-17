import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import requests
import json
import os
from typing import Optional

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")

class DeepSeekAPI:
    """
    DeepSeek API 调用类
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 DeepSeek API
        """
        self.api_key = 'DeepSeek API'
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set it as DEEPSEEK_API_KEY environment variable or pass it to the constructor.")
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, user_text: str, emotion: str, intensity: str, max_length: int = 100) -> str:
        """
        使用 DeepSeek API 生成回应
        """
        # 根据情绪类型构造不同的提示词
        emotion_prompts = {
            'neutral': "你是一个温和的倾听者。对方现在情绪平静，他说：",
            'calm': "你是一个平和的陪伴者。对方现在心情平静，他说：",
            'happy': "你是一个温暖的分享者。对方现在心情愉快，他说：",
            'sad': "你是一个温柔的安慰者。对方现在感到悲伤，他说：",
            'angry': "你是一个冷静的疏导者。对方现在感到愤怒，他说：",
            'fearful': "你是一个安心的守护者。对方现在感到害怕，他说：",
            'disgust': "你是一个理解的倾听者。对方现在感到厌恶，他说：",
            'surprised': "你是一个好奇的分享者。对方现在感到惊讶，他说："
        }

        # 构造完整的 Prompt
        base_prompt = emotion_prompts.get(emotion)
        prompt = f"""{base_prompt} "{user_text}" 
        请你像朋友一样，回复一句温柔,自然的话。不要说"我会说"、"你应该说"、"答案"，
        只说这句话本身，不要添加其他内容。20字以内。"""

        # 构造请求数据
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个温柔、善解人意的树洞。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_length
        }

        try:
            # 发送请求
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()  # 检查响应状态
            
            # 解析响应
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            return "抱歉，我现在无法回应，请稍后再试。"
        except (KeyError, IndexError) as e:
            print(f"解析响应错误: {e}")
            return "抱歉，响应格式有误，请稍后再试。"

class LanguageModel:
    """
    大语言模型类，使用Qwen-1.8B模型
    """
    def __init__(self, model_name="Qwen/Qwen-1_8B", use_deepseek: bool = False):
        """
        初始化语言模型
        """
        self.use_deepseek = use_deepseek
        
        if use_deepseek:
            self.model = DeepSeekAPI()
            print("使用 DeepSeek API 模式")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                # 使用全精度以获得更好的性能
                torch_dtype=torch.float32,
            )
            
            # 如果有GPU则使用GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            print(f"使用本地模型模式，已加载到设备: {self.device}")
    
    def generate_response(self, user_text, emotion, intensity, max_length=100):
        """
        根据用户文本和情绪生成回应
        """
        if self.use_deepseek:
            return self.model.generate_response(user_text, emotion, intensity, max_length)
        
        # 根据情绪类型构造不同的提示词
        emotion_prompts = {
            'neutral': "你是一个温和的倾听者。对方现在情绪平静，他说：",
            'calm': "你是一个平和的陪伴者。对方现在心情平静，他说：",
            'happy': "你是一个温暖的分享者。对方现在心情愉快，他说：",
            'sad': "你是一个温柔的安慰者。对方现在感到悲伤，他说：",
            'angry': "你是一个冷静的疏导者。对方现在感到愤怒，他说：",
            'fearful': "你是一个安心的守护者。对方现在感到害怕，他说：",
            'disgust': "你是一个理解的倾听者。对方现在感到厌恶，他说：",
            'surprised': "你是一个好奇的分享者。对方现在感到惊讶，他说："
        }

        base_prompt = emotion_prompts.get(emotion)
        prompt = f"""
        {base_prompt} "{user_text}"
        请你像朋友一样，回复一句温柔,自然的话。不要说"我会说"、"你应该说"、"答案"，
        只说这句话本身，不要添加其他内容。20字以内。
        """


        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成回应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )
        
        # 解码回应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回应（去除原始Prompt）
        response = response.replace(prompt, "").strip()
        
        return response

if __name__ == "__main__":
    # 示例用法
    # # 使用本地模型
    # lm_local = LanguageModel(use_deepseek=False)
    # 使用 DeepSeek API
    lm_api = LanguageModel(use_deepseek=False)
    
    user_text = "我每天都在笑，但其实很难受。"
    emotion = "sad"
    intensity = "high"
    
    # 测试本地模型
    # response_local = lm_local.generate_response(user_text, emotion, intensity)
    # print(f"本地模型生成的回应: {response_local}")
    
    # 测试 API 模型
    response_api = lm_api.generate_response(user_text, emotion, intensity)
    print(f"API 模型生成的回应: {response_api}")