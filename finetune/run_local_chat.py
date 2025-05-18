import os
import torch
import argparse
import time
import warnings
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from colorama import init, Fore, Style

# 初始化colorama
init()

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    """打印彩色文本"""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

def get_old_model_version(model_path):
    """尝试从适配器配置中获取原始基础模型信息"""
    try:
        config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(config_path):
            config = PeftConfig.from_pretrained(model_path)
            if hasattr(config, "base_model_name_or_path"):
                return config.base_model_name_or_path
    except:
        pass
    return None

def load_model(model_path, base_model="Qwen/Qwen-1_8B-Chat", use_4bit=False, cpu_offload=False):
    """加载微调好的模型"""
    print_color(f"正在加载模型: {model_path}", Fore.YELLOW)
    start_time = time.time()
    
    # 尝试使用合适的方式加载模型
    model = None
    tokenizer = None
    
    # 检查是否可以从适配器配置中找到原始基础模型
    original_base_model = get_old_model_version(model_path)
    if original_base_model and original_base_model != base_model:
        print_color(f"检测到模型是用 {original_base_model} 微调的", Fore.YELLOW)
        print_color(f"当前指定的基础模型是 {base_model}", Fore.YELLOW)
        print_color(f"将尝试使用原始基础模型: {original_base_model}", Fore.YELLOW)
        base_model_to_try = [original_base_model, base_model]
    else:
        base_model_to_try = [base_model]
    
    # 尝试不同的基础模型
    last_error = None
    for current_base_model in base_model_to_try:
        try:
            print_color(f"尝试使用基础模型: {current_base_model}", Fore.CYAN)
            
            # 加载tokenizer
            print_color("加载tokenizer...", Fore.CYAN)
            tokenizer = AutoTokenizer.from_pretrained(
                current_base_model, 
                trust_remote_code=True
            )
            
            # 尝试为早期版本的tokenizer设置必要的标记
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = "<|endoftext|>"
                    
            # 设置模型加载配置
            load_args = {
                "trust_remote_code": True,
            }
            
            # 使用非量化的正常模式
            if cpu_offload:
                print_color("启用CPU辅助模式...", Fore.CYAN)
                # 设置CPU辅助参数
                load_args["device_map"] = "auto"
                
                # 确保offload目录存在且正确设置
                offload_folder = "offload_folder"
                if not os.path.exists(offload_folder):
                    os.makedirs(offload_folder)
                load_args["offload_folder"] = offload_folder
                print_color(f"将使用 {offload_folder} 目录进行模型分层存储", Fore.CYAN)
            else:
                # 常规加载模式
                load_args["device_map"] = "auto"
            
            # 加载基础模型
            print_color("加载基础模型...", Fore.CYAN)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                current_base_model, 
                **load_args
            )
            
            # 为解决词汇表大小不匹配问题添加代码
            # 先尝试获取适配器的配置
            print_color("读取适配器配置...", Fore.CYAN)
            adapter_config = PeftConfig.from_pretrained(model_path)
            
            # 尝试加载适配器
            print_color(f"加载LoRA适配器: {model_path}", Fore.CYAN)
            try:
                # 尝试以正常方式加载
                model = PeftModel.from_pretrained(
                    base_model_obj, 
                    model_path, 
                    is_trainable=False,  # 设置为推理模式
                    config=adapter_config
                )
            except Exception as e:
                if "size mismatch" in str(e) and "weight" in str(e):
                    print_color("检测到词汇表大小不匹配，尝试调整适配器...", Fore.YELLOW)
                    
                    # 提取错误信息中的尺寸
                    size_pattern = r"torch.Size\(\[(\d+), (\d+)\]\)"
                    matches = re.findall(size_pattern, str(e))
                    
                    if len(matches) >= 2:
                        adapter_vocab_size = int(matches[0][0])
                        model_vocab_size = int(matches[1][0])
                        
                        print_color(f"适配器词汇表大小: {adapter_vocab_size}, 模型词汇表大小: {model_vocab_size}", Fore.YELLOW)
                        
                        # 如果模型词汇表更大，尝试调整模型的嵌入层
                        if model_vocab_size > adapter_vocab_size:
                            print_color("尝试调整模型以匹配适配器词汇表大小...", Fore.YELLOW)
                            # 调整词汇表大小
                            base_model_obj.resize_token_embeddings(adapter_vocab_size)
                            
                            # 再次尝试加载适配器
                            model = PeftModel.from_pretrained(
                                base_model_obj, 
                                model_path, 
                                is_trainable=False
                            )
                        else:
                            raise
                    else:
                        raise
                elif "Cannot copy out of meta tensor" in str(e):
                    print_color("检测到meta tensor错误，尝试使用替代加载方法...", Fore.YELLOW)
                    
                    # 处理meta tensor错误的替代方案
                    # 1. 先将基础模型移动到实际设备
                    if hasattr(base_model_obj, "half"):
                        base_model_obj = base_model_obj.half()  # 使用半精度节省显存
                        
                    # 2. 使用不同的方式加载适配器
                    from peft import get_peft_model
                    
                    # 创建新的LoRA配置
                    print_color("使用get_peft_model重新应用适配器...", Fore.YELLOW)
                    from peft import LoraConfig
                    
                    # 从适配器配置中提取LoRA配置
                    lora_config = LoraConfig(
                        r=adapter_config.r,
                        lora_alpha=adapter_config.lora_alpha,
                        target_modules=adapter_config.target_modules,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                    
                    # 应用LoRA配置到基础模型
                    model = get_peft_model(base_model_obj, lora_config)
                    
                    # 加载权重
                    adapter_state_dict = torch.load(os.path.join(model_path, "adapter_model.bin"), map_location="cpu")
                    model.load_state_dict(adapter_state_dict, strict=False)
                else:
                    raise
            
            # 将模型设置为评估模式
            model.eval()
            
            # 一旦成功，跳出循环
            break
            
        except Exception as e:
            print_color(f"使用 {current_base_model} 加载失败: {str(e)}", Fore.RED)
            last_error = e
            model = None
            tokenizer = None
    
    # 如果所有尝试都失败
    if model is None or tokenizer is None:
        error_msg = f"所有模型加载尝试均失败"
        if last_error:
            error_msg += f": {str(last_error)}"
            
        if "CUDA out of memory" in str(last_error) or "device_map" in str(last_error):
            print_color("GPU显存不足，请尝试以下解决方案:", Fore.YELLOW)
            print_color("1. 使用 --cpu_offload 参数启用CPU辅助", Fore.WHITE)
            print_color("2. 选择更小的基础模型 (例如 Qwen-1.8B-Chat)", Fore.WHITE)
            print_color("3. 减少最大生成长度 (例如 --max_length 128)", Fore.WHITE)
        elif "size mismatch" in str(last_error):
            print_color("词汇表大小不匹配，请尝试以下解决方案:", Fore.YELLOW)
            print_color("1. 尝试使用与微调时相同版本的基础模型", Fore.WHITE)
            print_color("2. 或使用 --cpu_offload 参数减少显存使用", Fore.WHITE)
        elif "Cannot copy out of meta tensor" in str(last_error):
            print_color("检测到meta tensor错误，请尝试以下解决方案:", Fore.YELLOW)
            print_color("1. 不要使用device_map='auto'，而是使用--cpu_offload参数", Fore.WHITE)
            print_color("2. 尝试安装最新版本的transformers和peft库", Fore.WHITE)
            print_color("3. 尝试使用不同的基础模型版本", Fore.WHITE)
        
        raise Exception(error_msg)
    
    # 计算加载时间
    load_time = time.time() - start_time
    print_color(f"模型加载完成! 耗时: {load_time:.2f}秒", Fore.GREEN)
    
    return model, tokenizer

def chat_with_model(model, tokenizer, emotion="neutral", use_history=True, max_length=512, typing_speed=0.01):
    """与模型进行对话"""
    # 情感系统提示映射
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
    print_color("\n当前系统角色:", Fore.MAGENTA)
    print_color(system_prompt, Fore.MAGENTA, Style.BRIGHT)
    print_color("\n" + "="*60, Fore.BLUE)
    print_color("微调后的Qwen本地对话系统 - 开始交流", Fore.CYAN, Style.BRIGHT)
    print_color("="*60, Fore.BLUE)
    
    chat_history = []
    
    while True:
        print_color("\n你: ", Fore.GREEN, end="")
        user_input = input()
        
        if user_input.lower() in ["exit", "quit", "q", "退出"]:
            break
            
        if user_input.lower() in ["clear", "c", "清空"]:
            chat_history = []
            print_color("对话历史已清空！", Fore.YELLOW)
            continue
        
        if user_input.lower() in ["help", "h", "帮助"]:
            print_help_info()
            continue
            
        if user_input.lower().startswith("emotion:"):
            try:
                new_emotion = user_input.split(":", 1)[1].strip().lower()
                if new_emotion in emotion_prompts:
                    emotion = new_emotion
                    system_prompt = emotion_prompts[emotion]
                    print_color(f"情感已切换为: {emotion}", Fore.YELLOW)
                    print_color(f"系统角色: {system_prompt}", Fore.MAGENTA)
                else:
                    print_color(f"未知情感类型: {new_emotion}", Fore.RED)
                    print_color(f"可用情感类型: {', '.join(emotion_prompts.keys())}", Fore.CYAN)
                continue
            except:
                print_color("情感切换格式错误，请使用 'emotion:类型' 格式", Fore.RED)
                continue
        
        # 构建提示
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 添加聊天历史
        if use_history:
            for turn in chat_history:
                prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
                prompt += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
            
        # 添加当前用户输入
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        
        # 生成回应
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # 确保输入数据在正确的设备上
            try:
                # 尝试将输入移动到模型的设备
                input_ids = inputs["input_ids"].to(model.device)
            except:
                # 如果失败，检查模型是否有device属性
                if hasattr(model, "hf_device_map"):
                    # 寻找模型的第一个设备
                    first_device = next(iter(model.hf_device_map.values()))
                    input_ids = inputs["input_ids"].to(first_device)
                else:
                    # 否则假设模型在CPU上
                    input_ids = inputs["input_ids"]
            
            print_color("思考中...", Fore.YELLOW, end="\r")
            
            with torch.no_grad():
                # 使用更稳定的生成配置
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码回应
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 提取模型回应
            response_marker = "<|im_start|>assistant\n"
            end_marker = "<|im_end|>"
            
            # 从完整输出中提取最后一个助手的回应
            assistant_responses = full_response.split(response_marker)
            last_response = assistant_responses[-1]
            
            if end_marker in last_response:
                model_response = last_response.split(end_marker)[0]
            else:
                model_response = last_response
                
            # 打印回应，逐字显示效果
            print_color("\n助手: ", Fore.BLUE, end="")
            model_response = model_response.strip()
            for char in model_response:
                print_color(char, Fore.WHITE, Style.BRIGHT, end="")
                time.sleep(typing_speed)  # 调整这个值以改变打印速度
            print()
            
            # 更新对话历史
            if use_history:
                chat_history.append({
                    "user": user_input,
                    "assistant": model_response
                })
        except Exception as e:
            print_color(f"\n生成回应时发生错误: {str(e)}", Fore.RED)
            print_color("请尝试重新提问或退出重新启动", Fore.YELLOW)

def print_help_info():
    """打印帮助信息"""
    print_color("\n" + "="*60, Fore.BLUE)
    print_color("帮助信息", Fore.CYAN, Style.BRIGHT)
    print_color("="*60, Fore.BLUE)
    print_color("- 输入 'exit', 'quit', 'q' 或 '退出'：结束对话", Fore.WHITE)
    print_color("- 输入 'clear', 'c' 或 '清空'：清除对话历史", Fore.WHITE)
    print_color("- 输入 'help', 'h' 或 '帮助'：显示此帮助信息", Fore.WHITE)
    print_color("- 输入 'emotion:类型'：切换情感状态", Fore.WHITE)
    print_color("\n可用情感类型:", Fore.CYAN)
    print_color("  neutral: 中性  calm: 平静  happy: 开心", Fore.WHITE)
    print_color("  sad: 悲伤  angry: 愤怒  fearful: 恐惧", Fore.WHITE)
    print_color("  disgust: 厌恶  surprised: 惊讶", Fore.WHITE)
    print_color("="*60, Fore.BLUE)

def main():
    parser = argparse.ArgumentParser(description="与微调后的Qwen模型对话")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="qwen_finetune_output",
        help="微调模型的路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen-1_8B-Chat",
        help="基础模型名称或路径"
    )
    parser.add_argument(
        "--emotion", 
        type=str, 
        default="neutral",
        choices=["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        help="初始情感状态"
    )
    parser.add_argument(
        "--cpu_offload", 
        action="store_true",
        help="启用CPU辅助，在显存不足时将部分模型加载到CPU"
    )
    parser.add_argument(
        "--no_history", 
        action="store_true",
        help="不使用对话历史记录（每次回答只看当前问题）"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="生成回答的最大长度"
    )
    parser.add_argument(
        "--typing_speed", 
        type=float, 
        default=0.01,
        help="模型回答的打字速度（数值越小越快）"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print_color(f"错误: 模型路径 {args.model_path} 不存在", Fore.RED)
        # 尝试检查上级目录
        parent_path = os.path.join("..", args.model_path)
        if os.path.exists(parent_path):
            args.model_path = parent_path
            print_color(f"找到模型路径: {args.model_path}", Fore.GREEN)
        else:
            print_color("请使用 --model_path 参数指定正确的模型路径", Fore.YELLOW)
            return
    
    # 加载模型
    try:
        model, tokenizer = load_model(
            args.model_path, 
            args.base_model, 
            False,  # 不使用4bit量化
            args.cpu_offload
        )
    except Exception as e:
        print_color(f"加载模型失败: {str(e)}", Fore.RED)
        return
    
    print_color("\n" + "="*60, Fore.BLUE)
    print_color("微调后的Qwen模型对话系统", Fore.CYAN, Style.BRIGHT)
    print_color("="*60, Fore.BLUE)
    print_color("\n指令:", Fore.YELLOW)
    print_color("- 输入 'exit', 'quit', 'q' 或 '退出' 结束对话", Fore.WHITE)
    print_color("- 输入 'clear', 'c' 或 '清空' 清除对话历史", Fore.WHITE)
    print_color("- 输入 'help', 'h' 或 '帮助' 显示帮助信息", Fore.WHITE)
    print_color("- 输入 'emotion:类型' 切换情感状态，例如 'emotion:sad'", Fore.WHITE)
    print_color("\n开始对话！输入问题即可与模型交流...", Fore.GREEN)
    
    # 开始对话
    try:
        chat_with_model(
            model, 
            tokenizer, 
            args.emotion, 
            not args.no_history,
            args.max_length,
            args.typing_speed
        )
    except KeyboardInterrupt:
        print_color("\n用户中断，正在退出...", Fore.YELLOW)
    except Exception as e:
        print_color(f"\n发生错误: {str(e)}", Fore.RED)
    finally:
        print_color("\n对话已结束，谢谢使用！", Fore.GREEN)

if __name__ == "__main__":
    main() 