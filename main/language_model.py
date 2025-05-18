import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import requests
import json
import os
import re
from typing import Optional
from peft import PeftModel, PeftConfig

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
    大语言模型类，使用微调后的Qwen模型
    """
    def __init__(self, model_name="Qwen/Qwen-1_8B", use_deepseek: bool = False):
        """
        初始化语言模型
        """
        self.use_deepseek = use_deepseek
        self.direct_model = None  # 用于直接访问的基础模型
        
        if use_deepseek:
            self.model = DeepSeekAPI()
            print("使用 DeepSeek API 模式")
        else:
            # 设置模型路径
            base_model = "Qwen/Qwen-1_8B-Chat"
            finetune_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finetune", "qwen_finetune_output")
            generation_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finetune", "generation_model")
            adjusted_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finetune", "adjusted_base_model")
            
            # 确保一致的数据类型
            torch_dtype = torch.float32
            
            # 首先尝试加载生成模型（优先使用）
            if os.path.exists(generation_model_path) and os.path.isdir(generation_model_path):
                print(f"找到生成模型: {generation_model_path}")
                try:
                    # 直接使用生成模型进行初始化
                    self.model = AutoModelForCausalLM.from_pretrained(
                        generation_model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                    )
                    # 加载tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                    if self.tokenizer.pad_token is None:
                        if self.tokenizer.eos_token is not None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        else:
                            self.tokenizer.pad_token = "<|endoftext|>"
                    
                    # 设置设备
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.direct_model = self.model  # 直接使用该模型作为生成模型
                    
                    print("成功加载生成模型，准备就绪")
                    self.model.eval()
                    return
                except Exception as e:
                    print(f"加载生成模型失败，将尝试加载调整后的基础模型: {str(e)}")
            
            print(f"加载微调模型: {finetune_model_path}")
            
            # 如果生成模型加载失败，回退到调整后的基础模型
            if os.path.exists(adjusted_model_path) and os.path.isdir(adjusted_model_path):
                print(f"找到已调整的基础模型: {adjusted_model_path}")
                try:
                    # 加载已调整的基础模型
                    base_model_obj = AutoModelForCausalLM.from_pretrained(
                        adjusted_model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                    )
                    print("成功加载已调整的基础模型")
                except Exception as e:
                    print(f"加载已调整模型失败，将使用原始基础模型: {str(e)}")
                    # 回退到加载原始基础模型
                    base_model_obj = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                    )
            else:
                # 加载原始基础模型
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "<|endoftext|>"
            
            # 如果有GPU则使用GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            try:
                # 加载适配器
                adapter_config = PeftConfig.from_pretrained(finetune_model_path)
                
                # 检查适配器的词汇表大小与基础模型是否匹配
                try:
                    adapter_state_dict = torch.load(os.path.join(finetune_model_path, "adapter_model.bin"), map_location="cpu")
                    
                    # 在适配器状态字典中寻找词汇表大小信息
                    adapter_vocab_size = None
                    for key in adapter_state_dict.keys():
                        if "wte.weight" in key:
                            adapter_vocab_size = adapter_state_dict[key].shape[0]
                            break
                    
                    # 获取基础模型的词汇表大小
                    model_vocab_size = base_model_obj.get_input_embeddings().weight.shape[0]
                    
                    if adapter_vocab_size and adapter_vocab_size != model_vocab_size:
                        print(f"检测到词汇表大小不匹配: 适配器={adapter_vocab_size}, 模型={model_vocab_size}")
                        print("调整基础模型以匹配适配器词汇表大小...")
                        # 调整基础模型的词汇表大小以匹配适配器
                        base_model_obj.resize_token_embeddings(adapter_vocab_size)
                        
                        # 保存调整后的基础模型
                        print(f"保存调整后的基础模型到: {adjusted_model_path}")
                        if not os.path.exists(adjusted_model_path):
                            try:
                                # 确保我们只保存基础模型，不包含LoRA适配器
                                if hasattr(base_model_obj, "save_pretrained"):
                                    # 保存模型
                                    base_model_obj.save_pretrained(adjusted_model_path)
                                    print("已成功保存调整后的基础模型，下次将直接加载")
                                    
                                    # 保存词汇表
                                    if hasattr(self.tokenizer, "save_pretrained"):
                                        vocab_path = os.path.join(adjusted_model_path, "tokenizer")
                                        self.tokenizer.save_pretrained(vocab_path)
                                        print(f"已保存词汇表到: {vocab_path}")
                            except Exception as save_e:
                                print(f"保存调整后的基础模型失败: {str(save_e)}")
                except Exception as e:
                    print(f"检查词汇表大小时出错，继续尝试: {str(e)}")
                
                # 加载适配器
                self.model = PeftModel.from_pretrained(
                    base_model_obj, 
                    finetune_model_path,
                    is_trainable=False,
                    config=adapter_config
                )
                print("成功加载微调模型适配器")
            except Exception as e:
                print(f"加载适配器失败: {str(e)}")
                
                # 如果失败，尝试使用特定的词汇表大小进行修复
                if "size mismatch" in str(e) and "weight" in str(e):
                    try:
                        # 从错误消息中提取尺寸信息
                        import re
                        size_pattern = r"torch.Size\(\[(\d+), (\d+)\]\)"
                        matches = re.findall(size_pattern, str(e))
                        
                        if len(matches) >= 2:
                            adapter_vocab_size = int(matches[0][0])
                            model_vocab_size = int(matches[1][0])
                            
                            print(f"从错误消息中检测到词汇表大小不匹配:")
                            print(f"适配器词汇表大小: {adapter_vocab_size}, 模型词汇表大小: {model_vocab_size}")
                            
                            # 调整基础模型的词汇表大小
                            print("尝试调整模型以匹配适配器词汇表大小...")
                            base_model_obj.resize_token_embeddings(adapter_vocab_size)
                            
                            # 保存调整后的基础模型
                            print(f"保存调整后的基础模型到: {adjusted_model_path}")
                            if not os.path.exists(adjusted_model_path):
                                try:
                                    # 确保我们只保存基础模型，不包含LoRA适配器
                                    if hasattr(base_model_obj, "save_pretrained"):
                                        # 保存模型
                                        base_model_obj.save_pretrained(adjusted_model_path)
                                        print("已成功保存调整后的基础模型，下次将直接加载")
                                        
                                        # 保存词汇表
                                        if hasattr(self.tokenizer, "save_pretrained"):
                                            vocab_path = os.path.join(adjusted_model_path, "tokenizer")
                                            self.tokenizer.save_pretrained(vocab_path)
                                            print(f"已保存词汇表到: {vocab_path}")
                                except Exception as save_e:
                                    print(f"保存调整后的基础模型失败: {str(save_e)}")
                            
                            # 再次尝试加载适配器
                            self.model = PeftModel.from_pretrained(
                                base_model_obj,
                                finetune_model_path,
                                is_trainable=False
                            )
                            print("修复后成功加载微调模型适配器")
                            self.model.eval()  # 确保模型处于评估模式
                            return
                    except Exception as nested_e:
                        print(f"尝试修复词汇表大小不匹配失败: {str(nested_e)}")
                
                # 硬编码修复已知的词汇表大小问题 (151851 vs 151936)
                if "151851" in str(e) and "151936" in str(e):
                    try:
                        print("尝试使用硬编码的词汇表大小进行修复...")
                        base_model_obj.resize_token_embeddings(151851)
                        
                        # 保存调整后的基础模型
                        print(f"保存调整后的基础模型到: {adjusted_model_path}")
                        if not os.path.exists(adjusted_model_path):
                            try:
                                # 确保我们只保存基础模型，不包含LoRA适配器
                                if hasattr(base_model_obj, "save_pretrained"):
                                    # 保存模型
                                    base_model_obj.save_pretrained(adjusted_model_path)
                                    print("已成功保存调整后的基础模型，下次将直接加载")
                                    
                                    # 保存词汇表
                                    if hasattr(self.tokenizer, "save_pretrained"):
                                        vocab_path = os.path.join(adjusted_model_path, "tokenizer")
                                        self.tokenizer.save_pretrained(vocab_path)
                                        print(f"已保存词汇表到: {vocab_path}")
                            except Exception as save_e:
                                print(f"保存调整后的基础模型失败: {str(save_e)}")
                        
                        # 再次尝试加载适配器
                        self.model = PeftModel.from_pretrained(
                            base_model_obj,
                            finetune_model_path,
                            is_trainable=False
                        )
                        print("使用硬编码词汇表大小成功加载微调模型适配器")
                        self.model.eval()  # 确保模型处于评估模式
                        return
                    except Exception as hardcode_e:
                        print(f"尝试硬编码修复失败: {str(hardcode_e)}")
                
                # 如果所有修复尝试都失败，则使用基础模型
                print("所有修复尝试均失败，使用基础模型")
                self.model = base_model_obj
            
            self.model.eval()
            print(f"模型已加载到设备: {self.device}")
    
    def generate_response(self, user_text, emotion, intensity, max_length=100):
        """
        根据用户文本和情绪生成回应
        """
        if self.use_deepseek:
            return self.model.generate_response(user_text, emotion, intensity, max_length)
        
        # 将中文情绪标签转换为英文(如果是中文的话)
        emotion_zh_to_en = {
            '中性': 'neutral',
            '平静': 'calm',
            '开心': 'happy',
            '悲伤': 'sad',
            '愤怒': 'angry',
            '恐惧': 'fearful',
            '厌恶': 'disgust',
            '惊讶': 'surprised'
        }
        
        if emotion in emotion_zh_to_en:
            emotion = emotion_zh_to_en[emotion]
        
        # 根据情绪类型构造不同的提示词
        emotion_prompts = {
            'neutral': "你是一个温和的倾听者。对方现在情绪平静。你会给予客观、平和的回应。",
            'calm': "你是一个平和的陪伴者。对方现在心情平静。你会提供平静、安宁的交流。",
            'happy': "你是一个温暖的分享者。对方现在心情愉快。你会用积极、欢快的语调回应。",
            'sad': "你是一个温柔的安慰者。对方现在感到悲伤。你会给予理解、安慰和支持。",
            'angry': "你是一个冷静的疏导者。对方现在感到愤怒。你会帮助他们平静下来并理性思考。",
            'fearful': "你是一个安心的守护者。对方现在感到害怕。你会提供安全感和鼓励。",
            'disgust': "你是一个理解的倾听者。对方现在感到厌恶。你会不带评判地理解他们的感受。",
            'surprised': "你是一个好奇的分享者。对方现在感到惊讶。你会帮助他们探索这种新体验。"
        }

        system_prompt = emotion_prompts.get(emotion, emotion_prompts['neutral'])
        
        # 添加更明确的指令，生成完整句子
        system_prompt += " 请用完整的句子回应，确保句子有明确的结束标点。回应应该温暖有同理心，不超过50个字。"
        
        # 构建Qwen的聊天格式，适用于微调后的模型
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"

        # 如果已经有直接模型，则使用它
        base_model = self.direct_model
        
        try:
            # 编码输入，确保包含attention_mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # 如果没有attention_mask，手动创建
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # 将张量移动到合适的设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # 进行生成
            with torch.no_grad():
                # 确保模型和输入在同一设备上
                model_device = next(self.model.parameters()).device
                input_ids = input_ids.to(model_device)
                attention_mask = attention_mask.to(model_device)
                
                # 使用模型生成
                print("使用模型进行文本生成...")
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # 解码回应
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # 提取生成的回应
                response_marker = "<|im_start|>assistant\n"
                end_marker = "<|im_end|>"
                
                # 从完整输出中提取最后一个助手的回应
                assistant_responses = full_response.split(response_marker)
                last_response = assistant_responses[-1]
                
                if end_marker in last_response:
                    model_response = last_response.split(end_marker)[0]
                else:
                    model_response = last_response
                    
                # 清理响应文本
                cleaned_text = model_response.strip()
                
                # 确保返回完整句子，最多50个字符
                response = ""
                char_count = 0
                sentence_end = False
                
                for char in cleaned_text:
                    response += char
                    char_count += 1
                    
                    # 检查是否到达句子结束
                    if char in "。！？.!?":
                        sentence_end = True
                        # 如果已经有超过20个字符，且遇到句子结束标记，结束提取
                        if char_count > 20:
                            break
                    
                    # 如果超过50个字符且没有找到句子结束，在合适的位置添加句号
                    if char_count >= 50 and not sentence_end:
                        # 在词语结束处添加句号
                        if char in "，,、 ":
                            response = response[:-1] + "。"
                            break
                        # 如果接近限制还没找到好的断句点，直接添加句号
                        elif char_count >= 60:
                            response += "。"
                            break
                        
                # 如果没有句号结尾，添加句号
                if not response.endswith(("。", "！", "？", ".", "!", "?")):
                    response += "。"
                    
                return response.strip()
                
        except Exception as e:
            print(f"生成过程中发生错误: {str(e)}")
            # 提供一个备用回应
            return "我在听着呢，请继续说吧。"

    def get_base_model(self):
        """获取可以直接用于生成的基础模型"""
        if self.direct_model is not None:
            return self.direct_model
            
        # 如果没有预先创建的基础模型，尝试创建一个
        try:
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "generate"):
                return self.model.base_model
            elif hasattr(self.model, "model") and hasattr(self.model.model, "generate"):
                return self.model.model
                
            return self.model
        except Exception as e:
            print(f"获取基础模型失败: {str(e)}")
            return None

if __name__ == "__main__":
    # 示例用法
    # 使用微调模型
    lm = LanguageModel(model_name="Qwen/Qwen-1_8B-Chat", use_deepseek=False)
    
    user_text = "我每天都在笑，但其实很难受。"
    emotion = "sad"
    intensity = "high"
    
    # 测试模型
    response = lm.generate_response(user_text, emotion, intensity)
    print(f"模型生成的回应: {response}")