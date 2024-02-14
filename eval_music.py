import torch
from audioldm_eval import EvaluationHelper
import os
from tqdm import tqdm
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--generation_result_path",
        type=str,
        required=False,
        help="Audio sampling rate during evaluation",
        default="/mnt/fast/datasets/audio/audioset/2million_audioset_wav/balanced_train_segments",
    )

    parser.add_argument(
        "-t",
        "--target_audio_path",
        type=str,
        required=False,
        help="Audio sampling rate during evaluation",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/musiccaps/musicap",
    )

    args = parser.parse_args()

    # GPU acceleration is preferred
    device = torch.device(f"cuda:{0}")

    generation_result_path = args.generation_result_path
    target_audio_path = args.target_audio_path
    tempt_dir = f'tempt_dir{random.randint(1, 999999)}/'

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
    try:
        metrics = evaluator.main(
            tempt_dir,
            target_audio_path,
            limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
        )
    except Exception as e:
        print(e)
        os.system(f"rm -r {tempt_dir}")
