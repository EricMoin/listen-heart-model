import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch.nn.functional as F
from torch.cuda.amp import autocast
import bitsandbytes as bnb
from numpy.core.multiarray import _reconstruct
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import platform
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CyclicLR
from pytorch_lightning.tuner.tuning import Tuner

# 全局设置微软雅黑字体
def setup_chinese_font():
    # 设置全局字体为微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置默认字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 验证字体是否可用
    try:
        font = FontProperties(family='Microsoft YaHei')
    except:
        print("警告: 无法设置微软雅黑字体，将使用系统默认字体")
        # 使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Arial Unicode MS']

# 设置全局字体
setup_chinese_font()

# 设置matplotlib其他样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 添加安全的全局函数
torch.serialization.add_safe_globals([_reconstruct, np.ndarray])

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 创建Rich控制台
console = Console()

class TrainingVisualizationCallback(pl.callbacks.Callback):
    """
    自定义回调以可视化训练过程
    """
    def __init__(self, save_dir='logs/plots'):
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.train_emotion_accs = []
        self.val_emotion_accs = []
        self.train_intensity_accs = []
        self.val_intensity_accs = []
        self.epochs = []
        self.current_epoch = -1
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用全局字体设置
        self.font = FontProperties(family='Microsoft YaHei')
    
    def on_train_epoch_start(self, trainer, pl_module):
        # 更新当前epoch
        self.current_epoch = trainer.current_epoch
        if self.current_epoch not in self.epochs:
            self.epochs.append(self.current_epoch)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # 收集训练指标
        train_loss = trainer.callback_metrics['train_loss'].item()
        train_emotion_acc = trainer.callback_metrics['train_emotion_acc'].item()
        train_intensity_acc = trainer.callback_metrics['train_intensity_acc'].item()
        
        self.train_losses.append(train_loss)
        self.train_emotion_accs.append(train_emotion_acc)
        self.train_intensity_accs.append(train_intensity_acc)
        
        # 创建epoch信息表格
        table = Table(title=f"Epoch {self.current_epoch} 训练指标", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right", style="green")
        
        table.add_row("训练损失", f"{train_loss:.4f}")
        table.add_row("情绪准确率", f"{train_emotion_acc:.2%}")
        table.add_row("强度准确率", f"{train_intensity_acc:.2%}")
        
        console.print(table)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # 收集验证指标
        val_loss = trainer.callback_metrics['val_loss'].item()
        val_emotion_acc = trainer.callback_metrics['val_emotion_acc'].item()
        val_intensity_acc = trainer.callback_metrics['val_intensity_acc'].item()
        
        self.val_losses.append(val_loss)
        self.val_emotion_accs.append(val_emotion_acc)
        self.val_intensity_accs.append(val_intensity_acc)
        
        # 创建验证指标表格
        table = Table(title=f"Epoch {self.current_epoch} 验证指标", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right", style="green")
        
        table.add_row("验证损失", f"{val_loss:.4f}")
        table.add_row("情绪准确率", f"{val_emotion_acc:.2%}")
        table.add_row("强度准确率", f"{val_intensity_acc:.2%}")
        
        console.print(table)
        console.print()  # 添加空行
    
    def on_train_end(self, trainer, pl_module):
        # 在训练结束时绘制最终图表
        self._plot_and_save("final")
    
    def _plot_and_save(self, epoch):
        # 确保所有列表长度一致
        min_length = min(len(self.epochs), len(self.train_losses), len(self.val_losses),
                        len(self.train_emotion_accs), len(self.val_emotion_accs),
                        len(self.train_intensity_accs), len(self.val_intensity_accs))
        
        if min_length == 0:
            return
        
        # 截取相同长度的数据
        epochs = self.epochs[:min_length]
        train_losses = self.train_losses[:min_length]
        val_losses = self.val_losses[:min_length]
        train_emotion_accs = self.train_emotion_accs[:min_length]
        val_emotion_accs = self.val_emotion_accs[:min_length]
        train_intensity_accs = self.train_intensity_accs[:min_length]
        val_intensity_accs = self.val_intensity_accs[:min_length]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 设置图表样式
        fig.patch.set_facecolor('white')
        
        # 绘制损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_title('训练和验证损失', pad=15, fontproperties=self.font)
        ax1.set_xlabel('Epoch', fontproperties=self.font)
        ax1.set_ylabel('损失', fontproperties=self.font)
        ax1.legend(prop=self.font)
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        ax2.plot(epochs, train_emotion_accs, 'g-', label='情绪训练准确率', linewidth=2)
        ax2.plot(epochs, val_emotion_accs, 'm-', label='情绪验证准确率', linewidth=2)
        ax2.plot(epochs, train_intensity_accs, 'c-', label='强度训练准确率', linewidth=2)
        ax2.plot(epochs, val_intensity_accs, 'y-', label='强度验证准确率', linewidth=2)
        ax2.set_title('训练和验证准确率', pad=15, fontproperties=self.font)
        ax2.set_xlabel('Epoch', fontproperties=self.font)
        ax2.set_ylabel('准确率', fontproperties=self.font)
        ax2.legend(prop=self.font)
        ax2.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.save_dir, f'training_curves_{timestamp}.png'))
        plt.close()

class EmotionDataset(Dataset):
    """
    情绪数据集类，用于加载和处理特征数据
    """
    def __init__(self, features, emotion_labels, intensity_labels, max_time_steps=100):
        self.features = features
        self.emotion_labels = emotion_labels
        self.intensity_labels = intensity_labels
        self.max_time_steps = max_time_steps  # 最大时间步长
        
        # 验证数据
        self._validate_data()
    
    def _validate_data(self):
        """
        验证数据格式是否正确
        """
        if len(self.features) != len(self.emotion_labels) or len(self.features) != len(self.intensity_labels):
            raise ValueError("特征和标签长度不匹配")
        
        # 检查第一个样本的格式
        sample = self.features[0]
        if not isinstance(sample, dict):
            raise ValueError("特征数据必须是字典格式")
        if 'mel_features' not in sample or 'wav2vec_features' not in sample:
            raise ValueError("特征数据必须包含 'mel_features' 和 'wav2vec_features'")
    
    def _process_mel_features(self, mel_features):
        """
        处理MEL特征，确保固定长度
        """
        # 转换为张量
        mel_tensor = torch.FloatTensor(mel_features)
        
        # 获取当前时间步长
        current_time_steps = mel_tensor.size(1)
        
        if current_time_steps > self.max_time_steps:
            # 如果太长，从中间截取
            start = (current_time_steps - self.max_time_steps) // 2
            mel_tensor = mel_tensor[:, start:start + self.max_time_steps]
        elif current_time_steps < self.max_time_steps:
            # 如果太短，在两侧填充
            pad_left = (self.max_time_steps - current_time_steps) // 2
            pad_right = self.max_time_steps - current_time_steps - pad_left
            mel_tensor = F.pad(mel_tensor, (pad_left, pad_right), mode='replicate')
        
        return mel_tensor
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        """
        try:
            feature = self.features[idx]
            
            # 确保特征是正确的格式
            if not isinstance(feature, dict):
                raise ValueError(f"特征数据格式错误: {type(feature)}")
            
            # 处理MEL特征
            mel_feature = self._process_mel_features(feature['mel_features'])
            wav2vec_feature = torch.FloatTensor(feature['wav2vec_features'])
            
            # 获取标签
            emotion_label = torch.LongTensor([self.emotion_labels[idx]])
            intensity_label = torch.LongTensor([self.intensity_labels[idx]])
            
            # 验证特征维度
            if mel_feature.size(1) != self.max_time_steps:
                raise ValueError(f"MEL特征时间步长错误: {mel_feature.size(1)} != {self.max_time_steps}")
            if wav2vec_feature.dim() != 1:
                raise ValueError(f"wav2vec特征维度错误: {wav2vec_feature.shape}")
            
            return {
                'mel': mel_feature,
                'wav2vec': wav2vec_feature
            }, emotion_label, intensity_label
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            raise e

class EmotionRecognitionModel(pl.LightningModule):
    """
    情绪识别模型，使用1D-CRNN结构
    """
    def __init__(self, input_dim, hidden_dim, emotion_classes, intensity_classes):
        super(EmotionRecognitionModel, self).__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 添加训练阶段控制
        self.current_stage = 'warmup'  # warmup, cyclic, cosine
        self.warmup_epochs = 30
        self.cyclic_epochs = 90
        self.cosine_epochs = 30
        
        # 添加学习率查找器支持
        self.lr_finder = None
        self.learning_rate = None
        
        # 添加SWA支持
        self.swa_model = None
        self.swa_start = None
        
        # 添加训练数据加载器长度属性
        self.train_dataloader_len = None
        
        # 添加进度条
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # 设置默认数据类型为全精度
        # 注意：不能直接设置dtype属性，而是应该使用to()方法转换数据类型
        
        # MEL特征的CNN层
        self.mel_cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=8, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=8, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout1d(0.3)
        )
        
        # wav2vec特征的处理层
        self.wav2vec_proj = nn.Linear(768, 256)  # wav2vec2-base输出维度为768
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 使用梯度检查点 - 修复实现
        def checkpoint_lstm(*args, **kwargs):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.lstm),
                *args,
                **kwargs
            )
        
        self.checkpoint_lstm = checkpoint_lstm
        
        # 情绪分类层
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, emotion_classes)
        )
        
        # 情绪强度分类层
        self.intensity_classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, intensity_classes)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def setup(self, stage=None):
        """
        在训练开始前设置数据加载器长度
        """
        if stage == 'fit':
            # 获取训练数据加载器的长度
            if self.trainer and self.trainer.train_dataloader:
                self.train_dataloader_len = len(self.trainer.train_dataloader)
            else:
                # 如果trainer还未初始化，使用一个合理的默认值
                self.train_dataloader_len = 100  # 这个值会在训练开始时更新
    
    def forward(self, x):
        # 分别处理MEL特征和wav2vec特征
        mel_features = x['mel']
        wav2vec_features = x['wav2vec']
        
        # 确保数据类型一致
        mel_features = mel_features.to(torch.float32)
        wav2vec_features = wav2vec_features.to(torch.float32)
        
        # 使用自动混合精度
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            # 处理MEL特征
            mel_out = self.mel_cnn(mel_features)
            mel_out = mel_out.transpose(1, 2)  # 调整维度顺序
            
            # 处理wav2vec特征
            wav2vec_out = self.wav2vec_proj(wav2vec_features)
        
            # 特征融合
            combined = torch.cat([mel_out, wav2vec_out.unsqueeze(1).expand(-1, mel_out.size(1), -1)], dim=-1)
            fused = self.fusion(combined)
        
            # BiLSTM处理 
            lstm_out, _ = self.checkpoint_lstm(fused)
            lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        
            # 情绪分类
            emotion_output = self.emotion_classifier(lstm_out)
        
            # 情绪强度分类
            intensity_output = self.intensity_classifier(lstm_out)
        
        return emotion_output, intensity_output

    def training_step(self, batch, batch_idx):
        features, emotion_labels, intensity_labels = batch
        emotion_output, intensity_output = self(features)
        
        # 计算损失
        emotion_loss = self.criterion(emotion_output, emotion_labels.squeeze())
        intensity_loss = self.criterion(intensity_output, intensity_labels.squeeze())
        loss = emotion_loss + 0.5 * intensity_loss
        
        # 计算准确率
        emotion_preds = torch.argmax(emotion_output, dim=1)
        intensity_preds = torch.argmax(intensity_output, dim=1)
        emotion_acc = (emotion_preds == emotion_labels.squeeze()).float().mean()
        intensity_acc = (intensity_preds == intensity_labels.squeeze()).float().mean()
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True, 
                on_step=False, on_epoch=True, 
                logger=True, sync_dist=True)
        self.log('train_emotion_acc', emotion_acc, prog_bar=True,
                on_step=False, on_epoch=True,
                logger=True, sync_dist=True)
        self.log('train_intensity_acc', intensity_acc, prog_bar=True,
                on_step=False, on_epoch=True,
                logger=True, sync_dist=True)
        
        # 保存输出用于epoch结束时的计算
        self.training_step_outputs.append({
            'loss': loss,
            'emotion_acc': emotion_acc,
            'intensity_acc': intensity_acc
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, emotion_labels, intensity_labels = batch
        emotion_output, intensity_output = self(features)
        
        # 计算损失
        emotion_loss = self.criterion(emotion_output, emotion_labels.squeeze())
        intensity_loss = self.criterion(intensity_output, intensity_labels.squeeze())
        loss = emotion_loss + 0.5 * intensity_loss
        
        # 计算准确率
        emotion_preds = torch.argmax(emotion_output, dim=1)
        intensity_preds = torch.argmax(intensity_output, dim=1)
        emotion_acc = (emotion_preds == emotion_labels.squeeze()).float().mean()
        intensity_acc = (intensity_preds == intensity_labels.squeeze()).float().mean()
        
        # 记录指标
        self.log('val_loss', loss, prog_bar=True,
                on_step=False, on_epoch=True,
                logger=True, sync_dist=True)
        self.log('val_emotion_acc', emotion_acc, prog_bar=True,
                on_step=False, on_epoch=True,
                logger=True, sync_dist=True)
        self.log('val_intensity_acc', intensity_acc, prog_bar=True,
                on_step=False, on_epoch=True,
                logger=True, sync_dist=True)
        
        # 保存输出用于epoch结束时的计算
        self.validation_step_outputs.append({
            'loss': loss,
            'emotion_acc': emotion_acc,
            'intensity_acc': intensity_acc
        })
        
        return loss
    
    def on_train_epoch_start(self):
        """在每个训练epoch开始时更新训练阶段"""
        current_epoch = self.current_epoch
        
        if current_epoch < self.warmup_epochs:
            self.current_stage = 'warmup'
        elif current_epoch < (self.warmup_epochs + self.cyclic_epochs):
            self.current_stage = 'cyclic'
        else:
            self.current_stage = 'cosine'
            
            # 在cosine阶段，如果达到SWA开始点，更新模型
            if self.swa_model is not None and current_epoch >= self.swa_start:
                self.swa_model.update_parameters(self)
    
    def on_train_epoch_end(self):
        """在每个训练epoch结束时更新SWA模型"""
        if self.current_stage == 'cosine' and self.swa_model is not None:
            if self.current_epoch >= self.swa_start:
                self.swa_model.update_parameters(self)
        
        # 原有的epoch结束代码
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_emotion_acc = torch.stack([x['emotion_acc'] for x in self.training_step_outputs]).mean()
        avg_intensity_acc = torch.stack([x['intensity_acc'] for x in self.training_step_outputs]).mean()
        
        self.log('epoch_train_loss', avg_loss, prog_bar=True)
        self.log('epoch_train_emotion_acc', avg_emotion_acc, prog_bar=True)
        self.log('epoch_train_intensity_acc', avg_intensity_acc)
        
        self.training_step_outputs.clear()
    
    def on_train_end(self):
        """在训练结束时更新SWA模型的BN统计信息"""
        if self.swa_model is not None:
            torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.swa_model)
            # 将SWA模型的状态复制到当前模型
            self.load_state_dict(self.swa_model.state_dict())

    def configure_optimizers(self):
        # 获取模型参数组
        param_groups = self._get_param_groups()
        
        # 使用8-bit Adam优化器
        optimizer = bnb.optim.Adam8bit(
            param_groups,
            lr=1e-3,
            weight_decay=0.01
        )
        
        # 获取实际的steps_per_epoch
        if self.train_dataloader_len is None:
            steps_per_epoch = 100
        else:
            steps_per_epoch = self.train_dataloader_len
        
        # 根据当前训练阶段选择不同的学习率调度器
        if self.current_stage == 'warmup':
            # 线性预热阶段
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs * steps_per_epoch
            )
        elif self.current_stage == 'cyclic':
            # CyclicLR阶段
            scheduler = CyclicLR(
                optimizer,
                base_lr=1e-4,
                max_lr=1e-3,
                step_size_up=steps_per_epoch * 5,  # 5个epoch上升
                step_size_down=steps_per_epoch * 5,  # 5个epoch下降
                mode='triangular2',
                cycle_momentum=False
            )
        else:  # cosine阶段
            # Cosine退火 + SWA
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.cosine_epochs * steps_per_epoch,
                eta_min=1e-6
            )
            
            # 初始化SWA
            if self.swa_model is None:
                self.swa_model = AveragedModel(self)
                self.swa_start = self.cosine_epochs // 2  # 在cosine阶段的一半处开始SWA
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def _get_param_groups(self):
        """获取分层参数组，用于分层冻结"""
        # 定义不同层的参数组
        param_groups = [
            # CNN层参数组
            {'params': self.mel_cnn.parameters(), 'lr': 1e-3},
            # wav2vec投影层参数组
            {'params': self.wav2vec_proj.parameters(), 'lr': 1e-3},
            # 特征融合层参数组
            {'params': self.fusion.parameters(), 'lr': 1e-3},
            # BiLSTM层参数组
            {'params': self.lstm.parameters(), 'lr': 1e-3},
            # 分类器层参数组
            {'params': self.emotion_classifier.parameters(), 'lr': 1e-3},
            {'params': self.intensity_classifier.parameters(), 'lr': 1e-3}
        ]
        
        # 根据训练阶段调整学习率
        if self.current_stage == 'warmup':
            # 在预热阶段，降低所有层的学习率
            for group in param_groups:
                group['lr'] *= 0.1
        
        return param_groups

class MemoryUsageCallback(pl.callbacks.Callback):
    """
    自定义回调以监控GPU内存使用情况
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available() and batch_idx % 100 == 0:  # 每100个批次记录一次
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            trainer.logger.experiment.add_scalar('memory/allocated_MB', memory_allocated, trainer.global_step)
            trainer.logger.experiment.add_scalar('memory/reserved_MB', memory_reserved, trainer.global_step)

class CustomProgressBar(pl.callbacks.ProgressBar):
    """
    自定义进度条，美化输出格式
    """
    def __init__(self):
        super().__init__()
        self._enabled = True
    
    def enable(self):
        """启用进度条"""
        self._enabled = True
    
    def disable(self):
        """禁用进度条"""
        self._enabled = False
    
    def get_metrics(self, trainer, model):
        # 获取当前指标
        items = super().get_metrics(trainer, model)
        
        # 格式化指标
        if 'train_loss' in items:
            items['train_loss'] = f"{items['train_loss']:.4f}"
        if 'train_emotion_acc' in items:
            items['train_emotion_acc'] = f"{items['train_emotion_acc']:.2%}"
        if 'train_intensity_acc' in items:
            items['train_intensity_acc'] = f"{items['train_intensity_acc']:.2%}"
        if 'val_loss' in items:
            items['val_loss'] = f"{items['val_loss']:.4f}"
        if 'val_emotion_acc' in items:
            items['val_emotion_acc'] = f"{items['val_emotion_acc']:.2%}"
        if 'val_intensity_acc' in items:
            items['val_intensity_acc'] = f"{items['val_intensity_acc']:.2%}"
            
        return items
    
    def on_train_start(self, trainer, pl_module):
        """训练开始时启用进度条"""
        super().on_train_start(trainer, pl_module)
        self.enable()
    
    def on_train_end(self, trainer, pl_module):
        """训练结束时禁用进度条"""
        super().on_train_end(trainer, pl_module)
        self.disable()
    
    def on_validation_start(self, trainer, pl_module):
        """验证开始时启用进度条"""
        super().on_validation_start(trainer, pl_module)
        self.enable()
    
    def on_validation_end(self, trainer, pl_module):
        """验证结束时禁用进度条"""
        super().on_validation_end(trainer, pl_module)
        self.disable()

class RichModelCheckpoint(ModelCheckpoint):
    """
    使用Rich美化的模型检查点回调
    """
    def _save_checkpoint(self, trainer, filepath):
        # 保存模型
        super()._save_checkpoint(trainer, filepath)
        
        # 获取当前指标
        current = trainer.callback_metrics[self.monitor].item()
        best = self.best_model_score.item() if self.best_model_score is not None else 0
        
        # 创建简化版的消息
        content = Text()
        content.append(f"Epoch {trainer.current_epoch:02d}", style="bold cyan")
        content.append(f": {self.monitor} = {current:.4f}", style="bold green")
        content.append(f" (best {best:.4f})", style="bold magenta")
        content.append(" - Model saved", style="italic")
        
        # 创建面板
        panel = Panel(
            content,
            border_style="bright_blue",
            padding=(0, 1)
        )
        
        # 打印面板
        console.print(panel)

def train_model(model, train_loader, val_loader, device, epochs=150):
    """
    使用PyTorch Lightning训练模型，支持分阶段训练
    """
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # 设置回调
    checkpoint_callback = RichModelCheckpoint(
        monitor='val_emotion_acc',
        dirpath='checkpoints',
        filename='emotion-{epoch:02d}-{val_emotion_acc:.2f}',
        save_top_k=3,
        mode='max',
        verbose=False
    )
    
    # 创建可视化回调
    viz_callback = TrainingVisualizationCallback()
    
    # 设置训练器
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        strategy='auto',
        callbacks=[
            checkpoint_callback,
            pl.callbacks.ModelSummary(max_depth=2),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            MemoryUsageCallback(),
            viz_callback,
            CustomProgressBar()
        ],
        gradient_clip_val=1.0,  # 梯度裁剪
        accumulate_grad_batches=2,
        logger=pl.loggers.TensorBoardLogger(
            save_dir='logs',
            name='emotion_recognition'
        ),
        enable_progress_bar=True,
        log_every_n_steps=1
    )
    
    # 在训练开始前设置数据加载器长度
    model.train_dataloader_len = len(train_loader)
    
    # 使用 Tuner 进行学习率查找
    print("正在运行学习率查找器...")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        train_dataloaders=train_loader,
        min_lr=1e-6,
        max_lr=1e-2,
        num_training=100,
        mode='exponential'
    )
    
    # 获取建议的学习率
    suggested_lr = lr_finder.suggestion()
    print(f"建议的学习率: {suggested_lr}")
    
    # 更新模型的学习率
    model.learning_rate = suggested_lr
    
    # 训练模型
    with console.status("[bold green]Training model...", spinner="dots"):
        trainer.fit(model, train_loader, val_loader)
    
    # 加载最佳模型
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        content = Text()
        content.append("\n最佳模型信息:", style="bold cyan")
        content.append(f"\n模型路径: {best_model_path}", style="bold blue")
        content.append(f"\n最佳验证情绪准确率: {checkpoint_callback.best_model_score:.2%}", style="bold green")
        content.append(f"\n最终学习率: {suggested_lr}", style="bold yellow")
        
        panel = Panel(
            content,
            title="[bold]Training Complete[/bold]",
            border_style="bright_green",
            padding=(1, 2)
        )
        console.print(panel)
        
        best_model = EmotionRecognitionModel.load_from_checkpoint(
            best_model_path,
            input_dim=model.hparams.input_dim,
            hidden_dim=model.hparams.hidden_dim,
            emotion_classes=model.hparams.emotion_classes,
            intensity_classes=model.hparams.intensity_classes
        )
        best_model = best_model.to(device)
        return best_model
    
    return model

def prepare_data(features_df, batch_size=16, test_size=0.2, max_time_steps=100):
    """
    准备训练和验证数据
    """
    # 提取特征和标签
    features = features_df['features'].values
    
    # 验证特征格式
    if not isinstance(features[0], dict):
        raise ValueError("特征数据必须是字典格式，包含 'mel_features' 和 'wav2vec_features'")
    
    # 获取特征维度
    mel_dim = features[0]['mel_features'].shape[0]  # MEL频谱的mel维度
    wav2vec_dim = features[0]['wav2vec_features'].shape[0]  # wav2vec特征维度
    
    print(f"MEL特征维度: {mel_dim}")
    print(f"wav2vec特征维度: {wav2vec_dim}")
    print(f"最大时间步长: {max_time_steps}")
    
    # 编码标签
    emotion_encoder = LabelEncoder()
    intensity_encoder = LabelEncoder()
    
    emotion_labels = emotion_encoder.fit_transform(features_df['emotion'].values)
    intensity_labels = intensity_encoder.fit_transform(features_df['intensity'].values)
    
    # 划分训练集和验证集
    X_train, X_val, y_emotion_train, y_emotion_val, y_intensity_train, y_intensity_val = train_test_split(
        features, emotion_labels, intensity_labels, test_size=test_size, random_state=42, stratify=emotion_labels
    )
    
    # 创建数据集
    train_dataset = EmotionDataset(X_train, y_emotion_train, y_intensity_train, max_time_steps=max_time_steps)
    val_dataset = EmotionDataset(X_val, y_emotion_val, y_intensity_val, max_time_steps=max_time_steps)
    
    # 创建数据加载器，添加持久化工作进程
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, emotion_encoder, intensity_encoder, mel_dim

def save_model(model, emotion_encoder, intensity_encoder, model_path):
    """
    保存模型和标签编码器
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型和编码器
    torch.save({
        'model_state_dict': model.state_dict(),
        'emotion_classes': emotion_encoder.classes_,
        'intensity_classes': intensity_encoder.classes_,
        'hyper_parameters': model.hparams  # 保存超参数
    }, model_path)
    
    print(f"模型已保存到 {model_path}")

def load_model(model_path, input_dim, hidden_dim, force_input_dim=False):
    """
    加载模型和标签编码器
    
    Args:
        model_path: 模型文件路径
        input_dim: 输入维度（仅当force_input_dim=True时使用）
        hidden_dim: 隐藏层维度
        force_input_dim: 是否强制使用指定的input_dim
    """
    # 加载模型和编码器
    print("正在加载情绪识别模型...")
    try:
        # 直接使用完整模式加载
        checkpoint = torch.load(model_path, weights_only=False)
        
        # 重建编码器
        emotion_encoder = LabelEncoder()
        intensity_encoder = LabelEncoder()
        emotion_encoder.classes_ = checkpoint['emotion_classes']
        intensity_encoder.classes_ = checkpoint['intensity_classes']
        
        # 获取超参数
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            # 使用保存的input_dim，除非强制指定
            input_dim = input_dim if force_input_dim else hparams.input_dim
            hidden_dim = hparams.hidden_dim
            emotion_classes = hparams.emotion_classes
            intensity_classes = hparams.intensity_classes
            print(f"从检查点加载的模型参数:")
            print(f"- 输入维度: {hparams.input_dim}")
            print(f"- 隐藏层维度: {hparams.hidden_dim}")
            print(f"- 情绪类别数: {hparams.emotion_classes}")
            print(f"- 强度类别数: {hparams.intensity_classes}")
        else:
            emotion_classes = len(emotion_encoder.classes_)
            intensity_classes = len(intensity_encoder.classes_)
            print(f"使用默认参数:")
            print(f"- 输入维度: {input_dim}")
            print(f"- 隐藏层维度: {hidden_dim}")
            print(f"- 情绪类别数: {emotion_classes}")
            print(f"- 强度类别数: {intensity_classes}")
        
        # 创建模型
        model = EmotionRecognitionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            emotion_classes=emotion_classes,
            intensity_classes=intensity_classes
        )
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型参数，使用输入维度: {input_dim}")
        
        return model, emotion_encoder, intensity_encoder
        
    except Exception as e:
        raise RuntimeError(f"加载模型时出错: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    features_path = "../output/ravdess_features.pkl"
    model_path = "../models/emotion_model.pth"
    
    # 加载特征数据
    features_df = pd.read_pickle(features_path)
    
    # 准备数据
    train_loader, val_loader, emotion_encoder, intensity_encoder, input_dim = prepare_data(features_df)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    hidden_dim = 128
    emotion_classes = len(emotion_encoder.classes_)
    intensity_classes = len(intensity_encoder.classes_)
    
    model = EmotionRecognitionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        emotion_classes=emotion_classes,
        intensity_classes=intensity_classes
    ).to(device)
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, device)
    
    # 保存模型
    save_model(model, emotion_encoder, intensity_encoder, model_path)