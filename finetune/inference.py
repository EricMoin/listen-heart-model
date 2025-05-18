import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """加载微调好的模型"""
    print(f"正在加载模型从: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        # 如果内存有限，可以使用4bit量化推理
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"
        # )
    )
    
    return model, tokenizer

def chat_with_model(model, tokenizer, emotion="neutral"):
    """与模型进行对话"""
    # 情感系统提示映射
    emotion_prompts = {
        'neutral': "你是一个温和的倾听者。对方现在情绪平静。",
        'calm': "你是一个平和的陪伴者。对方现在心情平静。",
        'happy': "你是一个温暖的分享者。对方现在心情愉快。",
        'sad': "你是一个温柔的安慰者。对方现在感到悲伤。",
        'angry': "你是一个冷静的疏导者。对方现在感到愤怒。",
        'fearful': "你是一个安心的守护者。对方现在感到害怕。",
        'disgust': "你是一个理解的倾听者。对方现在感到厌恶。",
        'surprised': "你是一个好奇的分享者。对方现在感到惊讶。"
    }
    
    system_prompt = emotion_prompts.get(emotion, emotion_prompts['neutral'])
    print(f"\n系统角色: {system_prompt}")
    
    chat_history = []
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["exit", "quit", "q", "退出"]:
            break
            
        if user_input.lower() in ["clear", "c", "清空"]:
            chat_history = []
            print("对话历史已清空！")
            continue
            
        if user_input.lower().startswith("emotion:"):
            try:
                new_emotion = user_input.split(":", 1)[1].strip().lower()
                if new_emotion in emotion_prompts:
                    emotion = new_emotion
                    system_prompt = emotion_prompts[emotion]
                    print(f"情感已切换为: {emotion}")
                    print(f"系统角色: {system_prompt}")
                else:
                    print(f"未知情感类型: {new_emotion}")
                    print(f"可用情感类型: {', '.join(emotion_prompts.keys())}")
                continue
            except:
                print("情感切换格式错误，请使用 'emotion:类型' 格式")
                continue
        
        # 构建提示
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 添加聊天历史
        for turn in chat_history:
            prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
            
        # 添加当前用户输入
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        
        # 生成回应
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
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
            
        print(f"\n助手: {model_response.strip()}")
        
        # 更新对话历史
        chat_history.append({
            "user": user_input,
            "assistant": model_response.strip()
        })

def main():
    parser = argparse.ArgumentParser(description="与微调后的Qwen模型对话")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="qwen_finetune_output",
        help="微调模型的路径"
    )
    parser.add_argument(
        "--emotion", 
        type=str, 
        default="neutral",
        choices=["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        help="初始情感状态"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 {args.model_path} 不存在")
        # 尝试检查上级目录
        parent_path = os.path.join("..", args.model_path)
        if os.path.exists(parent_path):
            args.model_path = parent_path
            print(f"找到模型路径: {args.model_path}")
        else:
            print("请使用 --model_path 参数指定正确的模型路径")
            return
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    print("\n" + "="*50)
    print("微调后的Qwen模型对话系统")
    print("="*50)
    print("\n指令:")
    print("- 输入 'exit', 'quit', 'q' 或 '退出' 结束对话")
    print("- 输入 'clear', 'c' 或 '清空' 清除对话历史")
    print("- 输入 'emotion:类型' 切换情感状态，例如 'emotion:sad'")
    print("- 可用情感类型: neutral, calm, happy, sad, angry, fearful, disgust, surprised")
    
    # 开始对话
    chat_with_model(model, tokenizer, args.emotion)
    
    print("\n对话已结束，谢谢使用！")

if __name__ == "__main__":
    main() 