import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def load_audio(file_path, sr=22050):
    """
    加载音频文件并返回音频数据和采样率
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=sr)
        return audio_data, sample_rate
    except Exception as e:
        print(f"加载音频文件 {file_path} 时出错: {e}")
        return None, None


def extract_features(audio_data, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    从音频数据中提取MEL频谱特征
    """
    # 提取MEL频谱
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # 转换为分贝单位
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 标准化
    mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

    return mel_spec_norm


def load_wav2vec2():
    """
    加载预训练的wav2vec2模型
    """

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base",
        torch_dtype=torch.float32
    )

    # 使用全精度模型，不进行量化
    # 如果有GPU，将模型移至GPU
    if torch.cuda.is_available():
        model = model.to("cuda")

    return processor, model


def process_ravdess_dataset(data_path):
    """
    处理RAVDESS数据集并提取特征
    """
    print(f"正在处理数据集: {data_path}")
    features_list = []
    emotion_list = []
    intensity_list = []

    # 加载wav2vec2模型
    processor, wav2vec_model = load_wav2vec2()
    wav2vec_model.eval()

    # 遍历所有演员目录
    for actor_dir in tqdm(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        # 遍历该演员的所有音频文件
        for audio_file in os.listdir(actor_path):
            if not audio_file.endswith('.wav'):
                continue

            # 解析文件名获取情绪和强度信息
            emotion_code = audio_file.split('-')[2]
            intensity_code = audio_file.split('-')[3]

            # 获取完整文件路径
            audio_path = os.path.join(actor_path, audio_file)

            try:
                # 加载音频
                audio_data, sr = librosa.load(
                    audio_path, sr=16000)  # wav2vec2需要16kHz采样率

                # 提取MEL频谱特征
                mel_features = extract_features(audio_data)

                # 提取wav2vec2特征
                with torch.no_grad():
                    inputs = processor(
                        audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                    outputs = wav2vec_model(**inputs)
                    wav2vec_features = outputs.last_hidden_state.mean(
                        dim=1).squeeze().numpy()

                # 合并特征
                combined_features = {
                    'mel_features': mel_features,
                    'wav2vec_features': wav2vec_features
                }

                # 保存特征和标签
                features_list.append(combined_features)
                emotion_list.append(emotion_code)
                intensity_list.append(intensity_code)

            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")
                continue

    # 创建DataFrame
    features_df = pd.DataFrame({
        'features': features_list,
        'emotion': emotion_list,
        'intensity': intensity_list
    })

    return features_df


def save_features(df, output_path):
    """
    保存提取的特征到文件
    """
    df.to_pickle(output_path)
    print(f"特征已保存到 {output_path}")


def load_features(file_path):
    """
    从文件加载特征
    """
    return pd.read_pickle(file_path)


if __name__ == "__main__":
    data_path = "../data/ravdess"
    output_path = "../output/ravdess_features.pkl"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 处理数据集并提取特征
    features_df = process_ravdess_dataset(data_path)

    # 保存特征
    save_features(features_df, output_path)

    print(f"处理完成，共 {len(features_df)} 个样本")
