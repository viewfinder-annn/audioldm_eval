import torch
from audioldm_eval import EvaluationHelper, EvaluationHelperParallel
import torch.multiprocessing as mp

device = torch.device(f"cuda:{0}")

generation_result_path = "xxx"
# generation_result_path = "example/unpaired"
target_audio_path = "xxx"
generation_result_path = "/oss/CEPH_DATA/zhangjunan/musiccaps/infer/epoch-0005_step-0300000_loss-0.342955/"
target_audio_path = "/oss/CEPH_DATA/zhangjunan/musiccaps/audio/"

target_sample_rate = 16000
import librosa
import soundfile as sf
import os

from tqdm import tqdm

# resample_generation_result_path = generation_result_path.replace('/oss/CEPH_DATA/zhangjunan/musiccaps/infer', '/mnt/workspace/home/zhangjunan/audioldm_eval/audioldm_eval/test_musiccaps/infer')
# resample_target_audio_path = target_audio_path.replace('/oss/CEPH_DATA/zhangjunan/musiccaps/audio', '/mnt/workspace/home/zhangjunan/audioldm_eval/audioldm_eval/test_musiccaps/audio')
# os.makedirs(resample_generation_result_path, exist_ok=True)
# os.makedirs(resample_target_audio_path, exist_ok=True)

# for wav in tqdm(os.listdir(generation_result_path)):
#     if wav.endswith('.wav'):
#         audio, sr = librosa.load(generation_result_path + wav, sr=target_sample_rate, mono=True)
#         sf.write(resample_generation_result_path + wav, audio, target_sample_rate)

# for wav in tqdm(os.listdir(target_audio_path)):
#     if wav.endswith('.wav'):
#         audio, sr = librosa.load(target_audio_path + wav, sr=target_sample_rate, mono=True)
#         sf.write(resample_target_audio_path + wav, audio, target_sample_rate)

resample_generation_result_path = '/mnt/workspace/home/zhangjunan/audioldm_eval/audioldm_eval/test_musiccaps/infer/epoch-0008_step-0541000_loss-0.316784'
resample_target_audio_path = '/mnt/workspace/home/zhangjunan/audioldm_eval/audioldm_eval/test_musiccaps/audio'
## Single GPU

evaluator = EvaluationHelper(16000, device, backbone="mert")

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    resample_generation_result_path,
    resample_target_audio_path,
)

## Multiple GPUs

# if __name__ == '__main__':    
#     evaluator = EvaluationHelperParallel(16000, 2)
#     metrics = evaluator.main(
#         generation_result_path,
#         target_audio_path,
#     )