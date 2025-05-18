@echo off
chcp 65001 >nul
echo =================================
echo Qwen QLoRA 微调流程
echo =================================
echo.

REM 设置日志文件
set LOG_FILE=finetune_log_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt
set LOG_FILE=%LOG_FILE: =0%
echo 日志将保存到: %LOG_FILE%
echo 开始时间: %date% %time% > %LOG_FILE%

echo 步骤1: 安装依赖项
echo 步骤1: 安装依赖项 >> %LOG_FILE%
python install_requirements.py 
if %ERRORLEVEL% NEQ 0 (
    echo 安装依赖项失败，请检查错误信息。
    echo 安装依赖项失败，错误代码: %ERRORLEVEL% >> %LOG_FILE%
    pause
    exit /b %ERRORLEVEL%
)
echo 依赖项安装成功 >> %LOG_FILE%
echo.

echo 步骤2: 准备训练数据
echo 步骤2: 准备训练数据 >> %LOG_FILE%
python prepare_finetune_data.py 
if %ERRORLEVEL% NEQ 0 (
    echo 准备训练数据失败，请检查错误信息。
    echo 准备训练数据失败，错误代码: %ERRORLEVEL% >> %LOG_FILE%
    pause
    exit /b %ERRORLEVEL%
)
echo 训练数据准备成功 >> %LOG_FILE%
echo.

echo 步骤3: 开始微调模型 (这可能需要几个小时)
echo 步骤3: 开始微调模型 >> %LOG_FILE%
echo 微调开始时间: %date% %time% >> %LOG_FILE%

REM 检查文件是否存在
if not exist train_formatted_qa_data.jsonl (
    echo 错误: 未找到训练数据文件 train_formatted_qa_data.jsonl
    echo 错误: 未找到训练数据文件 train_formatted_qa_data.jsonl >> %LOG_FILE%
    echo 当前目录文件: >> %LOG_FILE%
    dir >> %LOG_FILE%
    pause
    exit /b 1
)

if not exist val_formatted_qa_data.jsonl (
    echo 错误: 未找到验证数据文件 val_formatted_qa_data.jsonl
    echo 错误: 未找到验证数据文件 val_formatted_qa_data.jsonl >> %LOG_FILE%
    echo 当前目录文件: >> %LOG_FILE%
    dir >> %LOG_FILE%
    pause
    exit /b 1
)

python finetune_qwen_qlora.py 
if %ERRORLEVEL% NEQ 0 (
    echo 微调模型失败，请检查错误信息。
    echo 微调模型失败，错误代码: %ERRORLEVEL% >> %LOG_FILE%
    pause
    exit /b %ERRORLEVEL%
)
echo 微调结束时间: %date% %time% >> %LOG_FILE%
echo 微调过程成功完成 >> %LOG_FILE%
echo.

echo 微调完成！模型已保存到 qwen_finetune_output 目录。
echo 可以使用以下命令测试微调后的模型:
echo python inference.py --model_path qwen_finetune_output
echo.
echo 完成时间: %date% %time% >> %LOG_FILE%
echo 全部流程成功结束 >> %LOG_FILE%
pause 