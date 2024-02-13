import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

generation_result_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/exp_result/exp_13_2_e15_musiccap_gs4/music/"
target_audio_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/musiccaps/musicap/"

# Initialize a helper instance
evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)