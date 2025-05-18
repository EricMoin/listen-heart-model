@echo off
chcp 65001 >nul
echo =================================
echo Qwen 微调模型快速启动
echo =================================
echo.

echo 使用优化配置启动聊天:
echo - 标准模式: 启用 (非量化模式，更高精度)
echo - CPU辅助: 启用 (解决显存不足问题)
echo - 减小最大输出长度: 128字符 (减少内存需求)
echo - 自动词汇表调整: 启用 (解决词汇表不匹配问题)
echo 启动中，模型加载可能需要几分钟...

python run_local_chat.py --model_path qwen_finetune_output --base_model Qwen/Qwen-1_8B-Chat  --max_length 128