import os
import pandas as pd
import json
from audiobox_aesthetics.infer import initialize_predictor
from tqdm import tqdm

def infer_from_folder(audio_src, csv_dst):
    # 初始化推理器
    predictor = initialize_predictor()
    
    # 获取文件夹中的所有音频文件路径
    audio_files = [os.path.join(audio_src, f) for f in os.listdir(audio_src) if f.endswith(('.wav', '.mp3'))]
    
    # 初始化结果列表
    results = []
    
    # 遍历音频文件并推理
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        res = predictor.forward([{"path": audio_path}])
        if res:
            result = res[0]
            result['path'] = audio_path  # 添加文件路径
            results.append(result)
    
    # 将结果保存到 DataFrame
    df = pd.DataFrame(results, columns=['path', 'CE', 'CU', 'PC', 'PQ'])
    
    # 输出到 CSV 文件
    df.to_csv(csv_dst, index=False)
    print(f"Results saved to {csv_dst}")
    
    # 计算四个指标的平均值并保存到JSON
    if not df.empty:
        avg_values = {
            'CE': round(df['CE'].mean(), 3),
            'CU': round(df['CU'].mean(), 3),
            'PC': round(df['PC'].mean(), 3),
            'PQ': round(df['PQ'].mean(), 3)
        }
        
        json_dst = csv_dst.replace(".csv", ".json")
        with open(json_dst, 'w') as f:
            json.dump(avg_values, f, indent=4)
        print(f"Average metrics saved to {json_dst}")

# 示例调用
if __name__ == "__main__":
    # audio_src = "/path/to/audio/folder"  # 替换为你的音频文件夹路径
    # csv_dst = "/path/to/output.csv"     # 替换为你的输出 CSV 文件路径
    
    audio_srcs = [
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_50hz_512_120chroma_0.1_drop_text_0.1_drop_vocal_use_tag/infer/musdb18-test/epoch-0009_step-0600000_loss--0.121251/accompaniment",
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_220M_[0.01]_0.1_drop_text_0.1_drop_vocal_use_tag/infer/musdb18-test/epoch-0007_step-0583000_loss--0.126835/accompaniment",
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/infer/fastsag/musdb18-test/accompaniment",
        
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/musdb18-test/accompaniment",
        
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_50hz_512_120chroma_0.1_drop_text_0.1_drop_vocal_use_tag/infer/yue-test/epoch-0009_step-0600000_loss--0.121251/accompaniment",
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_220M_[0.01]_0.1_drop_text_0.1_drop_vocal_use_tag/infer/yue-test/epoch-0007_step-0583000_loss--0.126835/accompaniment",
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/infer/fastsag/yue-test/accompaniment",
        
        "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/yue-test/accompaniment",
    ]
    
    for audio_src in audio_srcs:
        csv_dst = os.path.join(audio_src, "../", "audiobox_aesthetics.csv")
        infer_from_folder(audio_src, csv_dst)