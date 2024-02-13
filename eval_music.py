import torch
from audioldm_eval import EvaluationHelper
import os
from tqdm import tqdm

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

generation_result_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/exp_result/exp_13_2_e15_musiccap_gs4/music/"
target_audio_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/musiccaps/musicap/"
tempt_dir = 'tempt_dir'

# get same files
target_files = os.listdir(target_audio_path)
current_files = os.listdir(generation_result_path)
target_files = [s.split('.')[0] for s in target_files]
current_files = [s for s in current_files if s.endswith('.mp3') or s.endswith('.wav')]
shared_files = [s for s in current_files if s.split('.')[0] in target_files]
print(f'total file numbers: {len(shared_files)}')
os.makedirs(tempt_dir, exist_ok=True)
for i in tqdm(range(len(shared_files)), desc='copying'):
    filename = shared_files[i]
    os.system(f"cp {os.path.join(generation_result_path, filename)} {os.path.join(tempt_dir, filename)}")


# Initialize a helper instance
evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)

os.system(f"rm -r {tempt_dir}")
