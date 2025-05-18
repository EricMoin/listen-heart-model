import subprocess
import sys
import os

def install_requirements():
    """安装微调Qwen模型所需的依赖项"""
    requirements = [
        "torch>=2.0.1",
        "transformers>=4.33.1",
        "peft>=0.5.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.1",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "tqdm>=4.66.1",
    ]
    
    print("安装依赖项...")
    for req in requirements:
        print(f"安装 {req}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    print("所有依赖项安装完成！")

def check_gpu():
    """检查是否有可用的GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"发现 {gpu_count} 个可用GPU:")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("警告: 没有可用的GPU。微调可能会非常慢。")
            return False
    except ImportError:
        print("警告: 无法导入torch。请先安装PyTorch。")
        return False

def create_directories():
    """创建必要的目录"""
    dirs = [
        "qwen_finetune_output"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"已创建目录: {dir_path}")

if __name__ == "__main__":
    print("准备QLoRA微调环境...")
    
    # 安装依赖项
    install_requirements()
    
    # 检查GPU
    has_gpu = check_gpu()
    if not has_gpu:
        proceed = input("没有检测到GPU。继续进行微调可能会很慢。是否继续? (y/n): ")
        if proceed.lower() != 'y':
            print("已取消微调设置。")
            sys.exit(0)
    
    # 创建目录
    create_directories()
    
    print("\n设置完成！现在您可以运行以下命令来准备数据和开始微调:")
    print("1. python prepare_finetune_data.py")
    print("2. python finetune_qwen_qlora.py") 