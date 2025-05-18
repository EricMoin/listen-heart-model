import os
import shutil
import sys

def main():
    """删除调整后的模型缓存以重置模型加载过程"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    adjusted_model_path = os.path.join(project_dir, "finetune", "adjusted_base_model")
    
    if os.path.exists(adjusted_model_path):
        print(f"找到调整后的模型缓存: {adjusted_model_path}")
        confirm = input("确定要删除这个缓存吗? 这将使得下次运行时重新进行调整(y/n): ")
        
        if confirm.lower() in ['y', 'yes']:
            try:
                shutil.rmtree(adjusted_model_path)
                print(f"已成功删除模型缓存: {adjusted_model_path}")
            except Exception as e:
                print(f"删除失败: {str(e)}")
        else:
            print("操作已取消")
    else:
        print(f"未找到调整后的模型缓存: {adjusted_model_path}")
        print("不需要重置")
        
    print("\n完成。下次运行时将重新处理模型。")

if __name__ == "__main__":
    main() 