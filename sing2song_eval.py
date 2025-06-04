import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")
# Initialize a helper instance
evaluator = EvaluationHelper(16000, device, backbone="mert")

# generation_result_path = "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_50hz_512_120chroma_0.1_drop_text_0.1_drop_vocal_use_tag/infer/musdb18-test/epoch-0009_step-0600000_loss--0.121251/accompaniment"
# generation_result_path = "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_220M_[0.01]_0.1_drop_text_0.1_drop_vocal_use_tag/infer/musdb18-test/epoch-0007_step-0583000_loss--0.126835/accompaniment"
generation_result_path = "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/infer/fastsag/musdb18-test/accompaniment"
target_audio_path = "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/musdb18-test/accompaniment"
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)

generation_result_path_list = [
    "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_50hz_512_120chroma_0.1_drop_text_0.1_drop_vocal_use_tag/infer/yue-test/epoch-0009_step-0600000_loss--0.121251/accompaniment",
    "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/ckpts/sing2song/fmt_sing2song_220M_[0.01]_0.1_drop_text_0.1_drop_vocal_use_tag/infer/yue-test/epoch-0007_step-0583000_loss--0.126835/accompaniment",
    "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/infer/fastsag/yue-test/accompaniment",
]
target_audio_path = "/mnt/workspace/zhangjunan/SpeechGenerationYC-sing2song/data/yue-test/accompaniment"


for generation_result_path in generation_result_path_list:
    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )